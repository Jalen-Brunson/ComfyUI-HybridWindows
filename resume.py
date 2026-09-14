"""Plan, preserve, and save complete H3 video/latent continuation pairs."""
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch
import folder_paths
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from comfy_extras.nodes_video import SaveVideo

from .dependencies import _mmh3_module

Run = io.Custom('H3_HYBRID_RUN')
LATEST = 'Latest completed run'


def video_latents(frames):
    return 2 + 5 * ((frames - 5) // 17)


def video_frames(latents):
    return 5 + 17 * ((latents - 2) // 5)


def audio_at(latent_index):
    groups, remainder = divmod(latent_index, 5)
    frames = groups * 17 + sum((1, 4, 4, 4, 4)[:remainder])
    return round(frames * 40 / 24)


def source_file(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(folder_paths.base_path) / path
    return path.resolve()


def plan_run(total_frames, chunk_frames, overlap_frames, accepted_chunks, new_chunks):
    if chunk_frames < 22 or overlap_frames < 5 or any((n - 5) % 17 for n in (chunk_frames, overlap_frames)):
        raise ValueError('Window and overlap must use H3 frame counts: 5 + 17 × n.')
    stride = chunk_frames - overlap_frames
    if stride <= 0:
        raise ValueError('Overlap must be smaller than the window.')
    count = 1 + math.ceil(max(total_frames - chunk_frames, 0) / stride)
    if not 0 <= accepted_chunks < count:
        raise ValueError(f'Keep fewer than {count} chunks so there is a continuation to generate.')
    fresh = count - accepted_chunks
    if new_chunks > 0:
        fresh = min(fresh, new_chunks)
    start_window = max(accepted_chunks - 1, 0)
    start_frame = start_window * stride
    windows = fresh + int(accepted_chunks > 0)
    length = min(chunk_frames + (windows - 1) * stride, total_frames - start_frame)
    return dict(total_frames=total_frames, chunk_frames=chunk_frames, overlap_frames=overlap_frames,
                accepted_chunks=accepted_chunks, next_chunks=accepted_chunks + fresh,
                start_window=start_window, start_frame=start_frame, run_frames=length, windows=windows,
                pinned_frames=chunk_frames if accepted_chunks else 0,
                prefix_frames=accepted_chunks * stride + overlap_frames if accepted_chunks else 0)


class H3HybridRunPlan(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3HybridRunPlan', display_name='V2V Run Plan', category='sampling/hybrid/continuation',
            inputs=[io.String.Input('source_video', force_input=True),
                    io.String.Input('prompts', multiline=True, force_input=True),
                    io.Int.Input('accepted_chunks', default=0, min=0, tooltip='0 starts fresh. Otherwise keep this many completed chunks.'),
                    io.Int.Input('new_chunks', default=2, min=0, tooltip='New chunks to generate now. 0 generates all remaining chunks.'),
                    io.Int.Input('chunk_frames', default=243, min=22, step=17),
                    io.Int.Input('overlap_frames', default=39, min=5, step=17),
                    io.Float.Input('source_start_seconds', default=0., min=0.),
                    io.Int.Input('max_frames', default=0, min=0, tooltip='0 uses the remaining source. A positive value limits the full project, before chunking.')],
            outputs=[Run.Output('run'), io.Float.Output('run_start_seconds'), io.Int.Output('run_frames'),
                     io.Int.Output('windows'), io.Int.Output('start_window'), io.Int.Output('pinned_frames'),
                     io.Int.Output('prefix_frames'), io.String.Output('run_prompts'), io.String.Output('report')])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')

    @classmethod
    def execute(cls, source_video, prompts, accepted_chunks=0, new_chunks=2, chunk_frames=243,
                overlap_frames=39, source_start_seconds=0., max_frames=0):
        path = source_file(source_video)
        stat = path.stat()
        probe = subprocess.run(['ffprobe', '-v', 'error', '-show_entries',
            'stream=codec_type,duration:format=duration', '-of', 'json', str(path)],
            capture_output=True, text=True, check=True, timeout=30)
        info = json.loads(probe.stdout)
        stream = next((s for s in info['streams'] if s['codec_type'] == 'video'), None)
        if stream is None:
            raise ValueError('Select a source file containing video.')
        duration = float(stream.get('duration', info['format']['duration']))
        start = round(source_start_seconds * 24) / 24
        available = math.floor((duration - start) * 24 + 1e-5)
        if max_frames > 0:
            available = min(available, max_frames)
        if available < 5:
            raise ValueError('The selected source interval must contain at least five frames at 24 fps.')
        total = 5 + 17 * ((available - 5) // 17)
        run = plan_run(total, chunk_frames, overlap_frames, accepted_chunks, new_chunks)
        run.update(source_video=str(path), source_size=stat.st_size, source_mtime_ns=stat.st_mtime_ns,
                   source_start_seconds=start, fps=24.)
        texts = [p.strip() for p in prompts.split('|') if p.strip()]
        if len(texts) == 1:
            selected = texts * run['windows']
        else:
            selected = texts[run['start_window']:run['start_window'] + run['windows']]
            if len(selected) != run['windows']:
                raise ValueError('Write one prompt to repeat, or one | separated prompt per global chunk, starting at chunk 1.')
        report = (f"Keep {accepted_chunks} chunks / {run['prefix_frames']} frames; generate {run['next_chunks'] - accepted_chunks} new chunks.\n"
                  f"Source frames {run['start_frame']}–{run['start_frame'] + run['run_frames'] - 1}; {run['windows']} windows including the pinned continuation window when resuming.\n"
                  f"Next run: set accepted_chunks to {run['next_chunks']}. Project length: {total} frames at 24 fps.")
        return io.NodeOutput(run, start + run['start_frame'] / 24, run['run_frames'], run['windows'],
                             run['start_window'], run['pinned_frames'], run['prefix_frames'], ' | '.join(selected), report)


def runs_directory():
    return Path(folder_paths.get_output_directory()) / 'latents' / 'hybrid_windows'


def completed_runs():
    records = []
    for path in runs_directory().glob('*.json'):
        record = json.loads(path.read_text())
        # The dropdown chooses an indexed pair, never an arbitrary path.
        master = runs_directory() / (path.stem + '.pt')
        video = Path(folder_paths.get_output_directory()) / record['video']
        if master.is_file() and video.is_file():
            records.append({**record, 'id': path.stem})
    return sorted(records, key=lambda r: r['completed_at_ns'], reverse=True)


def run_label(record):
    return f"{record['project']} / {record['run']['next_chunks']} chunks / {record['video']} [{record['id']}]"


def resolve_run(project, choice):
    records = [r for r in completed_runs() if r['project'] == project]
    if choice == LATEST and records:
        return records[0]
    for record in records:
        if choice == run_label(record):
            return record
    raise ValueError('No matching completed run. Start with accepted_chunks = 0, or select a completed run from this project.')


class H3HybridResumeLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3HybridResumeLoad', display_name='Load V2V Continuation', category='sampling/hybrid/continuation',
            inputs=[Run.Input('run'), io.String.Input('project', force_input=True),
                    io.Combo.Input('saved_run', options=[LATEST] + [run_label(r) for r in completed_runs()], default=LATEST,
                                   tooltip='Latest is scoped to this project. Refresh ComfyUI to list newly saved runs.')],
            outputs=[io.Latent.Output('prior'), io.String.Output('prefix_video'), io.String.Output('report')])

    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return float('nan')

    @classmethod
    def execute(cls, run, project, saved_run=LATEST):
        if not run['accepted_chunks']:
            return io.NodeOutput(None, '', 'Fresh run: no saved video or master is loaded.')
        record = resolve_run(project, saved_run)
        for key in ('source_video', 'source_size', 'source_mtime_ns', 'source_start_seconds', 'chunk_frames', 'overlap_frames'):
            if record['run'][key] != run[key]:
                raise ValueError(f'Continuation {key} differs from the saved run. Restore the original setting or use a new project with accepted_chunks = 0.')
        if run['prefix_frames'] > record['frames']:
            raise ValueError(f"The selected run has {record['run']['next_chunks']} completed chunks; lower accepted_chunks.")
        saved = torch.load(runs_directory() / (record['id'] + '.pt'), weights_only=True, map_location='cpu')
        prior = {'samples': NestedTensor([saved['video'], saved['audio']])}
        video = (Path(folder_paths.get_output_directory()) / record['video']).resolve()
        video.relative_to(Path(folder_paths.get_output_directory()).resolve())
        return io.NodeOutput(prior, str(video), f"Keeping {run['accepted_chunks']} chunks from {record['id']}; prefix pixels use the paired saved video.")


class H3HybridResumeLatent(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3HybridResumeLatent', display_name='Prepare V2V and Inpainting', category='sampling/hybrid/continuation',
            inputs=[io.Latent.Input('source'), Run.Input('run'),
                    io.Boolean.Input('inpainting', default=False, tooltip='Enable the mask video: white regenerates, black preserves. Full overlap pinning is selected automatically.'),
                    io.Latent.Input('prior', optional=True, lazy=True), io.Mask.Input('mask', optional=True, lazy=True)],
            outputs=[io.Latent.Output('latent'), io.Mask.Output('denoise_mask'),
                     io.Combo.Output('overlap_pin', options=['full', 'tail_custom'])])

    @classmethod
    def check_lazy_status(cls, run, inpainting=False, prior=None, mask=None, **kwargs):
        return (['prior'] if run['accepted_chunks'] and prior is None else []) + (['mask'] if inpainting and mask is None else [])

    @classmethod
    def execute(cls, source, run, inpainting=False, prior=None, mask=None):
        video, audio = source['samples'].unbind()
        if video.shape[2] != video_latents(run['run_frames']):
            raise ValueError('Source encode length differs from the planned run. Check the source interval.')
        if inpainting and (mask is None or len(mask) not in (1, run['run_frames'])):
            raise ValueError('Load one mask per frame in this run, or a single static mask.')
        latent = source
        if run['accepted_chunks']:
            if prior is None:
                raise ValueError('Connect the saved continuation master.')
            pv, pa = prior['samples'].unbind()
            cut = run['start_frame'] // 17 * 5
            pinned = video_latents(run['pinned_frames'])
            if pv.shape[:2] + pv.shape[3:] != video.shape[:2] + video.shape[3:]:
                raise ValueError('The saved master and source must have the same output dimensions.')
            if pv.shape[2] < cut + pinned:
                raise ValueError('The saved master does not contain the accepted window.')
            video, audio = video.clone(), audio.clone()
            video[:, :, :pinned] = pv[:, :, cut:cut + pinned].to(video)
            a0 = audio_at(cut)
            length = min(audio_at(cut + pinned) - a0, audio_at(pinned), audio.shape[-1])
            if pa.shape[-1] < a0 + length:
                raise ValueError('The saved master has incomplete accepted audio.')
            audio[..., :length] = pa[..., a0:a0 + length].to(audio)
            latent = {'samples': NestedTensor([video, audio])}
        return io.NodeOutput(latent, mask if inpainting else None, 'full' if inpainting else 'tail_custom')


class H3HybridResumeOutput(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='H3HybridResumeOutput', display_name='Assemble V2V Continuation', category='sampling/hybrid/continuation',
            inputs=[io.Image.Input('images'), io.Audio.Input('audio'), Run.Input('run'),
                    io.Image.Input('prefix_images', optional=True, lazy=True), io.Audio.Input('prefix_audio', optional=True, lazy=True)],
            outputs=[io.Image.Output('images'), io.Audio.Output('audio'), io.String.Output('report')])

    @classmethod
    def check_lazy_status(cls, run, prefix_images=None, prefix_audio=None, **kwargs):
        if not run['accepted_chunks']:
            return []
        return (['prefix_images'] if prefix_images is None else []) + (['prefix_audio'] if prefix_audio is None else [])

    @classmethod
    def execute(cls, images, audio, run, prefix_images=None, prefix_audio=None):
        if len(images) != run['run_frames']:
            raise ValueError('Decoded video length differs from the planned run.')
        if not run['accepted_chunks']:
            return io.NodeOutput(images, audio, f'Fresh video: {len(images)} frames.')
        if prefix_images is None or len(prefix_images) != run['prefix_frames'] or prefix_audio is None:
            raise ValueError('Load exactly the accepted prefix frames and their audio before assembly.')
        return _mmh3_module('nodes_loop').MMH3JoinAV.execute(prefix_images, images, run['pinned_frames'], 0, prefix_audio, audio)


def merge_master(current, prior, run):
    video, audio = current['samples'].unbind()
    if run['start_frame']:
        pv, pa = prior['samples'].unbind()
        cut = run['start_frame'] // 17 * 5
        video = torch.cat((pv[:, :, :cut].to(video), video), dim=2)
        audio = torch.cat((pa[..., :audio_at(cut)].to(audio), audio), dim=-1)
    return video.detach().cpu(), audio.detach().cpu()


class H3HybridSaveRun(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        schema = SaveVideo.define_schema()
        schema.node_id = 'H3HybridSaveRun'
        schema.display_name = 'Save V2V Video and Continuation'
        schema.category = 'sampling/hybrid/continuation'
        schema.description = 'Save the complete video and its matching raw latent master. Only a fully saved pair becomes available for continuation.'
        schema.inputs.extend([io.Latent.Input('latent'), Run.Input('run'), io.String.Input('project', force_input=True),
                              io.Latent.Input('prior', optional=True, lazy=True)])
        schema.outputs.extend([io.String.Output('video_path'), io.String.Output('master_path'), io.String.Output('report')])
        return schema

    @classmethod
    def check_lazy_status(cls, run, prior=None, **kwargs):
        return ['prior'] if run['start_frame'] and prior is None else []

    @classmethod
    def execute(cls, video, filename_prefix, format, latent, run, project, codec=None, prior=None):
        if not project.strip():
            raise ValueError('Give this source and set of settings a project name.')
        if run['start_frame'] and prior is None:
            raise ValueError('A continuation save needs its prior master.')
        v, a = merge_master(latent, prior, run)
        frames = run['start_frame'] + run['run_frames']
        if video_frames(v.shape[2]) != frames or video.get_frame_count() != frames:
            raise ValueError('The assembled video and latent master must have the same full length.')
        directory = runs_directory()
        directory.mkdir(parents=True, exist_ok=True)
        stamp = time.time_ns()
        name = f"{hashlib.sha256(project.encode()).hexdigest()[:12]}_{stamp}"
        master = directory / (name + '.pt')
        temporary = directory / (name + '.pt.tmp')
        torch.save({'video': v, 'audio': a, 'start_chunk': 0,
                    'chunk_frames': run['chunk_frames'], 'overlap_frames': run['overlap_frames']}, temporary)
        os.replace(temporary, master)
        saved = SaveVideo.execute.__func__(cls, video, filename_prefix, format, codec)
        item = saved.ui.as_dict()['images'][0]
        output = Path(folder_paths.get_output_directory()).resolve()
        path = (output / item['subfolder'] / item['filename']).resolve()
        relative = path.relative_to(output)
        if not path.is_file():
            raise FileNotFoundError('The video did not finish saving; this run was not registered for continuation.')
        record = dict(project=project, video=str(relative), frames=frames, run=run, completed_at_ns=stamp)
        temporary = directory / (name + '.json.tmp')
        temporary.write_text(json.dumps(record, indent=2) + '\n')
        os.replace(temporary, directory / (name + '.json'))
        report = f"Saved {frames} frames. Next run: accepted_chunks = {run['next_chunks']}. Keep project '{project}' to continue."
        return io.NodeOutput(video, str(path), str(master), report, ui=saved.ui)
