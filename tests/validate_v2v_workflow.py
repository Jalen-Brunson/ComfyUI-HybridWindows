"""Validate the public workflow and execute its lazy branches with CPU fixtures."""
import asyncio
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_v2v_workflow import build, nodes, NAME
import execution
import folder_paths
import torch
from comfy.nested_tensor import NestedTensor

torch.set_num_threads(2)
calls, captured = [], []
resume = None


class FixtureRun:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'accepted': ('INT', {'default': 0})}}
    RETURN_TYPES = ('H3_HYBRID_RUN',)
    FUNCTION = 'run'
    def run(self, accepted): return (resume.plan_run(1263, 243, 39, accepted, 2 if accepted == 0 else 1),)


class FixtureLatent:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'run': ('H3_HYBRID_RUN',), 'prefix': ('BOOLEAN', {'default': False})}}
    RETURN_TYPES = ('LATENT',)
    FUNCTION = 'run'
    def run(self, run, prefix):
        calls.append('prior' if prefix else 'source')
        frames = run['prefix_frames'] if prefix else run['run_frames']
        value = .25 if prefix else .75
        return ({'samples': NestedTensor([torch.full((1, 24, resume.video_latents(frames), 2, 2), value),
                    torch.full((1, 32, 2, round(frames * 40 / 24)), value)])},)


class FixtureFrames:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'run': ('H3_HYBRID_RUN',), 'prefix': ('BOOLEAN', {'default': False})}}
    RETURN_TYPES = ('IMAGE', 'AUDIO')
    FUNCTION = 'run'
    def run(self, run, prefix):
        calls.append('prefix' if prefix else 'frames')
        count = run['prefix_frames'] if prefix else run['run_frames']
        value = .25 if prefix else .75
        return torch.full((count, 8, 8, 3), value), {'sample_rate': 240, 'waveform': torch.full((1, 2, count * 10), value)}


class FixtureMask:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'run': ('H3_HYBRID_RUN',)}}
    RETURN_TYPES = ('MASK',)
    FUNCTION = 'run'
    def run(self, run):
        calls.append('mask')
        return (torch.zeros(run['run_frames'], 8, 8),)


class FixtureOutput:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'images': ('IMAGE',), 'audio': ('AUDIO',), 'latent': ('LATENT',), 'pin': ('COMBO', {'options': ['full', 'tail_custom']})}}
    RETURN_TYPES = ()
    FUNCTION = 'run'
    OUTPUT_NODE = True
    def run(self, images, audio, latent, pin): captured.append((images, audio, latent, pin)); return ()


class FixtureBlurOutput:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'images': ('IMAGE',)}}
    RETURN_TYPES = ()
    FUNCTION = 'run'
    OUTPUT_NODE = True
    def run(self, images): captured.append(images); return ()


async def blur_executor_checks():
    nodes.NODE_CLASS_MAPPINGS['FixtureBlurOutput'] = FixtureBlurOutput
    results = []
    for enabled, strength in ((False, .5), (True, 0.), (True, .5)):
        graph = {
            '1': {'class_type': 'FixtureRun', 'inputs': {'accepted': 0}},
            '2': {'class_type': 'FixtureFrames', 'inputs': {'run': ['1', 0], 'prefix': False}},
            '3': {'class_type': 'FixtureMask', 'inputs': {'run': ['1', 0]}},
            '4': {'class_type': 'H3SegmentedVideoBlur', 'inputs': {'images': ['2', 0], 'enabled': enabled,
                'strength': strength, 'region_grow': 0, 'protect_grow': 0, 'min_component': .05,
                'region_mask': ['3', 0], 'protect_mask': ['3', 0]}},
            '5': {'class_type': 'FixtureBlurOutput', 'inputs': {'images': ['4', 0]}},
        }
        valid = await execution.validate_prompt('v2v-blur-lazy', graph, None)
        assert valid[0] and not valid[3], valid
        calls.clear(); captured.clear()
        server = SimpleNamespace(client_id=None, last_node_id=None, send_sync=lambda *a, **kw: None)
        runner = execution.PromptExecutor(server, cache_args={'ram': 0, 'ram_inactive': 0})
        await runner.execute_async(graph, 'v2v-blur-lazy', execute_outputs=valid[2])
        assert runner.success, runner.status_messages
        assert calls.count('mask') == bool(enabled and strength), calls
        results.append({'enabled': enabled, 'strength': strength, 'calls': calls.copy()})
    return results


async def executor_checks():
    for cls in (FixtureRun, FixtureLatent, FixtureFrames, FixtureMask, FixtureOutput):
        nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
    results = []
    for accepted in (0, 2):
        for inpainting in (False, True):
            graph = {
                '1': {'class_type': 'FixtureRun', 'inputs': {'accepted': accepted}},
                '2': {'class_type': 'FixtureLatent', 'inputs': {'run': ['1', 0], 'prefix': False}},
                '3': {'class_type': 'FixtureLatent', 'inputs': {'run': ['1', 0], 'prefix': True}},
                '4': {'class_type': 'FixtureMask', 'inputs': {'run': ['1', 0]}},
                '5': {'class_type': 'H3HybridResumeLatent', 'inputs': {'run': ['1', 0], 'source': ['2', 0], 'prior': ['3', 0], 'mask': ['4', 0], 'inpainting': inpainting}},
                '6': {'class_type': 'FixtureFrames', 'inputs': {'run': ['1', 0], 'prefix': False}},
                '7': {'class_type': 'FixtureFrames', 'inputs': {'run': ['1', 0], 'prefix': True}},
                '8': {'class_type': 'H3HybridResumeOutput', 'inputs': {'run': ['1', 0], 'images': ['6', 0], 'audio': ['6', 1], 'prefix_images': ['7', 0], 'prefix_audio': ['7', 1]}},
                '9': {'class_type': 'FixtureOutput', 'inputs': {'images': ['8', 0], 'audio': ['8', 1], 'latent': ['5', 0], 'pin': ['5', 2]}},
            }
            valid = await execution.validate_prompt('v2v-cpu', graph, None)
            assert valid[0] and not valid[3], valid
            calls.clear(); captured.clear()
            server = SimpleNamespace(client_id=None, last_node_id=None, send_sync=lambda *a, **kw: None)
            runner = execution.PromptExecutor(server, cache_args={'ram': 0, 'ram_inactive': 0})
            await runner.execute_async(graph, 'v2v-cpu', execute_outputs=valid[2])
            assert runner.success, runner.status_messages
            assert calls.count('prefix') == bool(accepted), calls
            assert calls.count('prior') == bool(accepted), calls
            assert calls.count('mask') == inpainting, calls
            images, audio, latent, pin = captured[0]
            assert pin == ('full' if inpainting else 'tail_custom')
            assert len(images) == (651 if accepted else 447)
            if accepted:
                assert torch.all(images[:447] == .25) and torch.all(images[447:] == .75)
                assert torch.all(audio['waveform'][..., :4470] == .25)
                assert torch.all(latent['samples'].unbind()[0][:, :, :72] == .25)
            results.append({'accepted_chunks': accepted, 'inpainting': inpainting, 'calls': calls.copy(), 'frames': len(images)})
    return results


reference_tokens = []


class ReferenceVae:
    def encode(self, frames):
        t = 1 if len(frames) == 1 else 2 + 5 * ((len(frames) - 5) // 17)
        return torch.full((1, 24, t, frames.shape[1] // 16, frames.shape[2] // 16), float(frames.mean()))


class ReferenceClip:
    def tokenize(self, text, minimax_ref_items=None):
        reference_tokens.append([item['type'] for item in minimax_ref_items])
        return text

    def encode_from_tokens_scheduled(self, text):
        return [[torch.zeros(1, 8), {'text': text}]]


class FixtureReferenceInputs:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {}}
    RETURN_TYPES = ('CLIP', 'VAE', 'IMAGE')
    FUNCTION = 'run'
    def run(self):
        control = torch.arange(39).ge(20).float().view(39, 1, 1, 1).expand(39, 64, 64, 3)
        return ReferenceClip(), ReferenceVae(), control


class FixtureConditioningOutput:
    @classmethod
    def INPUT_TYPES(cls): return {'required': {'cond_set': ('MMH3_COND_SET',)}}
    RETURN_TYPES = ()
    FUNCTION = 'run'
    OUTPUT_NODE = True
    def run(self, cond_set): captured.append(cond_set); return ()


async def reference_executor_checks(api):
    for cls in (FixtureReferenceInputs, FixtureConditioningOutput):
        nodes.NODE_CLASS_MAPPINGS[cls.__name__] = cls
    loader = nodes.NODE_CLASS_MAPPINGS['MiniMaxH3RefModsLoader']
    mods_module = importlib.import_module(loader.__module__)
    mp = importlib.import_module(nodes.NODE_CLASS_MAPPINGS['MMH3ReferenceMultiPrompt'].__module__)
    loader_inputs = dict(next(n['inputs'] for n in api.values() if n['class_type'] == 'MiniMaxH3RefModsLoader'))
    results = []
    with tempfile.TemporaryDirectory() as temp:
        mod = mods_module.H3RefMod('fixture', 'image', torch.full((1, 24, 1, 4, 4), .8125),
                                  description='a person with short dark hair', concept_type='identity')
        mod.save(str(Path(temp) / 'fixture'))
        Image.new('RGB', (64, 64), (128, 128, 128)).save(Path(temp) / 'portrait.png')
        loader_inputs['mod_1'] = 'fixture'
        with patch.object(mods_module, '_mod_search_dirs', return_value=[temp]), patch.object(folder_paths, 'get_input_directory', return_value=temp):
            for mode in (0, 2, 4):
                if mode == 2:
                    (Path(temp) / 'portrait.png').unlink()
                graph = {
                    '1': {'class_type': 'FixtureReferenceInputs', 'inputs': {}},
                    '2': {'class_type': 'LoadImage', 'inputs': {'image': 'portrait.png'}},
                    '3': {'class_type': 'MMH3ReferenceMultiPrompt', 'inputs': {
                        'clip': ['1', 0], 'vae': ['1', 1], 'audio_vae': ['1', 1], 'width': 64, 'height': 64,
                        'length': 39, 'ref_image_size': 'match', 'ref_images': ['2', 0],
                        'ref_videos.ref_video_0': ['1', 2], 'window_ref_video': True, 'chunk_frames': 22,
                        'overlap_frames': 5, 'use_input_audio': False, 'unload_text_encoder': False,
                        'prompts': ('The person in <Picture 1>' if mode == 0 else 'A person with short dark hair') + ' follows <Video 1>.'}},
                    '4': {'class_type': 'MiniMaxH3RefModsLoader', 'inputs': loader_inputs},
                    '5': {'class_type': 'H3RefModCondSetApply', 'inputs': {'cond_set': ['3', 0], 'mods': ['4', 0],
                        'retention': 1., 'insert_position': 'before_controls', 'controls_override': -1}},
                    '6': {'class_type': 'FixtureConditioningOutput', 'inputs': {'cond_set': ['5', 0]}},
                }
                if mode in (2, 4):
                    # Frontend export omits muted/bypassed LoadImage and its optional link.
                    del graph['2']; del graph['3']['inputs']['ref_images']
                valid = await execution.validate_prompt('v2v-refmod-only', graph, None)
                assert valid[0] and not valid[3], valid
                captured.clear(); reference_tokens.clear(); mp._CACHE.clear()
                server = SimpleNamespace(client_id=None, last_node_id=None, send_sync=lambda *a, **kw: None)
                runner = execution.PromptExecutor(server, cache_args={'ram': 0, 'ram_inactive': 0})
                await runner.execute_async(graph, 'v2v-refmod-only', execute_outputs=valid[2])
                assert runner.success, runner.status_messages
                assert reference_tokens == ([['image', 'video']] if mode == 0 else [['video']]) * 2, reference_tokens
                conds = captured[0]['conds']
                assert len(conds) == 2
                for index, cond in enumerate(conds):
                    refs = cond[0][1]['minimax_refs']
                    assert [b['kind'] for b in refs] == (['image', 'image', 'video'] if mode == 0 else ['image', 'video'])
                    torch.testing.assert_close(refs[-2]['latent'], mod.latent, rtol=0, atol=0)
                    expected = torch.arange(39).ge(20).float()[index * 17:index * 17 + 22].mean()
                    torch.testing.assert_close(refs[-1]['latent'].mean(), expected)
                results.append({'reference_mode': mode, 'windows': 2, 'refmod_in_every_window': True,
                                'tokenizer_reference_types': reference_tokens.copy()})
    return results


def check_links_and_layout(graph):
    by_id = {n['id']: n for n in graph['nodes']}
    for lid, source, slot, target, inp, kind in graph['links']:
        assert by_id[target]['inputs'][inp]['link'] == lid
        assert lid in by_id[source]['outputs'][slot]['links']
    for i, a in enumerate(graph['nodes']):
        x, y = a['pos']; w, h = a['size']
        for b in graph['nodes'][i+1:]:
            bx, by = b['pos']; bw, bh = b['size']
            assert not (x < bx+bw and bx < x+w and y-30 < by+bh and by-30 < y+h), (a['title'], b['title'])


def installed_model_paths(prompt):
    # Public downloads go in the documented root; this test machine uses subfolders.
    for node in prompt.values():
        if node['class_type'] == 'LoraLoaderModelOnly':
            name = node['inputs']['lora_name']
            choices = folder_paths.get_filename_list('loras')
            if name not in choices:
                node['inputs']['lora_name'] = next(p for p in choices if Path(p).name == name)
    return prompt


async def main():
    global resume
    with tempfile.TemporaryDirectory() as temp:
        source = Path(temp) / 'source.mp4'
        subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i', 'color=c=gray:s=32x32:r=24',
                        '-f', 'lavfi', '-i', 'anullsrc=r=32000:cl=stereo', '-t', '55', '-c:v', 'libx264',
                        '-preset', 'ultrafast', '-c:a', 'aac', str(source)], check=True)
        graph, api = await build(str(source), 'input/not_needed_when_disabled.webm', str(source))
        resume = importlib.import_module(nodes.NODE_CLASS_MAPPINGS['H3HybridRunPlan'].__module__)
        check_links_and_layout(graph)
        assert not any('InsightFace' in n['type'] or n['type'] == 'FaceAnonymizeVideo' for n in graph['nodes'])
        types = {n['class_type']: n for n in api.values()}
        assert sum(n['class_type'] == 'KSamplerAdvanced' for n in api.values()) == 2
        assert types['H3HybridWindows']['inputs']['context_frames'] == 5
        assert types['H3HybridWindows']['inputs']['accepted_prefix_frames'][1] == 5
        assert types['H3HybridWindows']['inputs']['start_window'][1] == 4
        blur_cls = nodes.NODE_CLASS_MAPPINGS['H3SegmentedVideoBlur']
        face = importlib.import_module(nodes.NODE_CLASS_MAPPINGS['FaceAnonymizeVideo'].__module__)
        image = (torch.arange(2 * 128 * 128 * 3).reshape(2, 128, 128, 3) % 256).float() / 255
        region = torch.zeros(2, 128, 128); region[:, 40:88, 40:88] = 1
        mouth = torch.zeros_like(region); mouth[:, 60:70, 56:74] = 1
        with patch.object(face, '_app_or_cpu', side_effect=AssertionError('InsightFace must not run')):
            blurred = blur_cls.execute(image, region_mask=region, protect_mask=mouth, region_grow=0, protect_grow=0)[0]
        torch.testing.assert_close(blurred[:, 60:70, 56:74], image[:, 60:70, 56:74], rtol=0, atol=0)
        torch.testing.assert_close(blurred[:, :15], image[:, :15], rtol=0, atol=0)
        assert (blurred[:, 44:54, 44:54] - image[:, 44:54, 44:54]).abs().mean() > .01
        assert blur_cls.execute(image, enabled=False)[0] is image
        assert blur_cls.check_lazy_status(enabled=False) == []
        assert blur_cls.check_lazy_status(strength=0.) == []
        cases = await executor_checks()
        with patch.object(face, '_app_or_cpu', side_effect=AssertionError('InsightFace must not run')):
            blur_cases = await blur_executor_checks()
        reference_cases = await reference_executor_checks(api)
        for inpaint in (False, True):
            for blur in (False, True):
                variant, prompt = await build(str(source), str(source), str(source), enable_mask=inpaint, enable_blur=blur, enable_control2=True)
                result = await execution.validate_prompt('v2v-public-schema', installed_model_paths(prompt), None)
                assert result[0] and not result[3], json.dumps(result, indent=2, default=str)
        for mode in (2, 4):
            variant, prompt = await build(str(source), str(source), str(source), reference_mode=mode)
            check_links_and_layout(variant)
            assert not any(n['class_type'] in ('LoadImage', 'MMH3ImageList') for n in prompt.values())
            conditioner = next(n for n in prompt.values() if n['class_type'] == 'MMH3ReferenceMultiPrompt')
            assert 'ref_images' not in conditioner['inputs']
            assert any(n['class_type'] == 'MiniMaxH3RefModsLoader' for n in prompt.values())
            result = await execution.validate_prompt('v2v-refmod-without-image-file', installed_model_paths(prompt), None)
            assert result[0] and not result[3], json.dumps(result, indent=2, default=str)
        result = await execution.validate_prompt('v2v-no-mask-file', installed_model_paths(api), None)
        assert result[0] and not result[3], json.dumps(result, indent=2, default=str)
        banned = ('vlm_video_prompt', 'wan_chunk_io', 'path_tools', 'minimax_h3_mask_tools', 'MaskVidExperiments')
        assert not any(any(name in module for name in banned) for module in sys.modules)
        report = {'workflow': NAME, 'processing_nodes': len(api), 'schema_variants': 7,
                  'layout_and_links': 'passed', 'segmentation_blur': 'passed without InsightFace',
                  'accepted_pixels_audio_and_lazy_branches': cases, 'lazy_blur': blur_cases, 'reference_modes': reference_cases, 'full_model_generation': 'not run'}
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
