"""Continuation planning, paired saves, exact accepted state and lazy assembly."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
COMFY = Path(os.environ.get('COMFYUI_ROOT', ROOT.parents[1]))
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], '--cpu']
import comfy.options
comfy.options.enable_args_parsing()
import torch
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io, ui
from comfy_extras.nodes_video import CreateVideo

spec = importlib.util.spec_from_file_location('hybrid_resume_test', ROOT / '__init__.py', submodule_search_locations=[str(ROOT)])
pack = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pack
spec.loader.exec_module(pack)
r = sys.modules[spec.name + '.resume']
torch.set_num_threads(2)


def latent(frames, value):
    return {'samples': NestedTensor([torch.full((1, 24, r.video_latents(frames), 2, 2), value),
                                     torch.full((1, 32, 2, round(frames * 40 / 24)), value)])}


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.output = patch.object(r.folder_paths, 'get_output_directory', return_value=self.temp.name)
        self.output.start()

    def tearDown(self):
        self.output.stop()
        self.temp.cleanup()

    def plan(self, accepted=0, new=2, total=1263):
        return {**r.plan_run(total, 243, 39, accepted, new), 'source_video': 'source.mp4', 'source_size': 7,
                'source_mtime_ns': 10, 'source_start_seconds': 0., 'fps': 24.}

    def save(self, run, current=None, prior=None, project='test'):
        frames = run['start_frame'] + run['run_frames']
        video = SimpleNamespace(get_frame_count=lambda: frames)
        filename = f'video_{len(r.completed_runs())}.mp4'
        (self.root / filename).write_bytes(b'complete video fixture')
        native = io.NodeOutput(video, ui=ui.PreviewVideo([ui.SavedResult(filename, '', io.FolderType.output)]))
        with patch.object(r.SaveVideo, 'execute', classmethod(lambda cls, *args: native)):
            out = r.H3HybridSaveRun.execute(video, 'video/test', 'mp4', current or latent(run['run_frames'], .5), run, project, prior=prior)
        return out

    def test_native_video_save_roundtrip_and_schema_after_core_registration(self):
        r.SaveVideo.GET_SCHEMA()
        r.H3HybridSaveRun.GET_SCHEMA()
        self.assertEqual(r.H3HybridSaveRun.RETURN_TYPES, ['VIDEO', 'STRING', 'STRING', 'STRING'])
        run = self.plan(0, 1, 22)
        frames = torch.full((22, 16, 16, 3), .5)
        video = CreateVideo.execute(frames, 24.)[0]
        saver = r.H3HybridSaveRun.PREPARE_CLASS_CLONE(None)
        saved = saver.execute(video, 'video/roundtrip', {'format': 'mp4', 'codec': {'codec': 'h264'}}, latent(22, .5), run, 'roundtrip')
        self.assertTrue(Path(saved[1]).is_file())
        self.assertEqual(r.completed_runs()[0]['frames'], 22)
        probe = r.subprocess.run(['ffprobe', '-v', 'error', '-count_frames', '-select_streams', 'v:0',
            '-show_entries', 'stream=nb_read_frames', '-of', 'json', saved[1]], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(probe.stdout)['streams'][0]['nb_read_frames'], '22')
        data = torch.load(saved[2], weights_only=True)
        torch.testing.assert_close(data['video'], latent(22, .5)['samples'].unbind()[0], rtol=0, atol=0)

    def test_fresh_and_capped_resume_geometry(self):
        fresh, resumed = self.plan(), self.plan(2)
        self.assertEqual((fresh['run_frames'], fresh['windows'], fresh['pinned_frames']), (447, 2, 0))
        self.assertEqual((resumed['start_frame'], resumed['run_frames'], resumed['windows']), (204, 651, 3))
        self.assertEqual((resumed['prefix_frames'], resumed['pinned_frames'], resumed['next_chunks']), (447, 243, 4))
        all_remaining = self.plan(2, 0, 906)
        self.assertEqual(all_remaining['run_frames'], 702)
        self.assertEqual(all_remaining['start_frame'] + all_remaining['run_frames'], 906)
        with self.assertRaises(ValueError):
            self.plan(6)
        with self.assertRaises(ValueError):
            r.plan_run(651, 243, 243, 0, 2)

    def test_source_probe_and_global_prompt_selection(self):
        path = self.root / 'source.mp4'; path.write_bytes(b'fixture')
        probe = SimpleNamespace(stdout=json.dumps({'streams': [{'codec_type': 'video', 'duration': '60'}], 'format': {'duration': '60'}}))
        with patch.object(r.subprocess, 'run', return_value=probe):
            out = r.H3HybridRunPlan.execute(str(path), 'first | second | third | fourth', 2, 2, source_start_seconds=1.01)
            self.assertEqual(out[7], 'second | third | fourth')
            self.assertEqual(out[1], 9.5)
            self.assertEqual(out[0]['source_start_seconds'], 1.)
            repeat = r.H3HybridRunPlan.execute(str(path), 'repeat', 2, 2)
            self.assertEqual(repeat[7], 'repeat | repeat | repeat')
            with self.assertRaisesRegex(ValueError, 'one prompt'):
                r.H3HybridRunPlan.execute(str(path), 'first | second', 2, 2)

    def test_accepted_master_and_new_source_regions_are_exact(self):
        run = self.plan(2)
        prior, source = latent(855, .25), latent(run['run_frames'], .75)
        out = r.H3HybridResumeLatent.execute(source, run, prior=prior)
        v, a = out[0]['samples'].unbind()
        self.assertTrue(torch.all(v[:, :, :72] == .25))
        self.assertTrue(torch.all(v[:, :, 72:] == .75))
        self.assertTrue(torch.all(a[..., :405] == .25))
        self.assertTrue(torch.all(a[..., 405:] == .75))
        self.assertTrue(torch.all(source['samples'].unbind()[0] == .75))
        self.assertEqual(out[2], 'tail_custom')
        self.assertIsNone(out[1])
        merged_v, merged_a = r.merge_master(out[0], prior, run)
        self.assertEqual(r.video_frames(merged_v.shape[2]), 855)
        self.assertTrue(torch.all(merged_v[:, :, :132] == .25))
        self.assertTrue(torch.all(merged_a[..., :745] == .25))

    def test_inpainting_selects_full_context_and_validates_alignment(self):
        run = self.plan()
        source = latent(run['run_frames'], .5)
        mask = torch.zeros(run['run_frames'], 32, 32)
        out = r.H3HybridResumeLatent.execute(source, run, True, mask=mask)
        self.assertIs(out[1], mask)
        self.assertEqual(out[2], 'full')
        with self.assertRaisesRegex(ValueError, 'mask per frame'):
            r.H3HybridResumeLatent.execute(source, run, True, mask=mask[:-1])
        self.assertEqual(r.H3HybridResumeLatent.check_lazy_status(run), [])
        self.assertEqual(r.H3HybridResumeLatent.check_lazy_status(run, True), ['mask'])
        self.assertEqual(r.H3HybridResumeLatent.check_lazy_status(self.plan(1), True), ['prior', 'mask'])

    def test_pair_selection_is_project_scoped_and_supports_older_runs(self):
        self.save(self.plan(), project='first')
        older = r.completed_runs()[0]
        self.save(self.plan(), project='second')
        self.save(self.plan(), project='first')
        latest = r.resolve_run('first', r.LATEST)
        self.assertNotEqual(latest['id'], older['id'])
        self.assertEqual(r.resolve_run('first', r.run_label(older))['id'], older['id'])
        with self.assertRaises(ValueError):
            r.resolve_run('second', r.run_label(older))
        with self.assertRaises(ValueError):
            r.resolve_run('first', '../../arbitrary.pt')

    def test_fresh_load_needs_no_existing_project(self):
        self.assertIsNone(r.H3HybridResumeLoad.execute(self.plan(), 'new')[0])
        self.assertEqual(r.H3HybridResumeOutput.check_lazy_status(self.plan()), [])
        self.assertEqual(r.H3HybridResumeOutput.check_lazy_status(self.plan(1)), ['prefix_images', 'prefix_audio'])

    def test_load_checks_source_grid_and_accepted_count(self):
        self.save(self.plan())
        out = r.H3HybridResumeLoad.execute(self.plan(2), 'test')
        self.assertEqual(out[0]['samples'].unbind()[0].shape[2], 132)
        with self.assertRaisesRegex(ValueError, 'lower accepted_chunks'):
            r.H3HybridResumeLoad.execute(self.plan(4), 'test')
        with self.assertRaisesRegex(ValueError, 'source_start_seconds'):
            r.H3HybridResumeLoad.execute({**self.plan(2), 'source_start_seconds': 2.}, 'test')
        with self.assertRaisesRegex(ValueError, 'overlap_frames'):
            r.H3HybridResumeLoad.execute({**self.plan(2), 'overlap_frames': 56}, 'test')

    def test_failed_save_and_missing_video_are_not_resumable(self):
        run = self.plan()
        video = SimpleNamespace(get_frame_count=lambda: 447)
        def fail(cls, *args):
            raise RuntimeError('encoder failed')
        with patch.object(r.SaveVideo, 'execute', classmethod(fail)):
            with self.assertRaisesRegex(RuntimeError, 'encoder failed'):
                r.H3HybridSaveRun.execute(video, 'video/test', 'mp4', latent(447, .5), run, 'test')
        self.assertEqual(r.completed_runs(), [])
        out = self.save(run)
        Path(out[1]).unlink()
        self.assertEqual(r.completed_runs(), [])

    def test_chained_save_keeps_full_master_and_redoes_only_suffix(self):
        self.save(self.plan(0, 2), latent(447, .25))
        run = self.plan(2, 2)
        prior = r.H3HybridResumeLoad.execute(run, 'test')[0]
        prepared = r.H3HybridResumeLatent.execute(latent(run['run_frames'], .75), run, prior=prior)[0]
        self.save(run, prepared, prior)
        again = self.plan(2, 1)
        prior2 = r.H3HybridResumeLoad.execute(again, 'test')[0]
        current2 = r.H3HybridResumeLatent.execute(latent(again['run_frames'], .9), again, prior=prior2)[0]
        saved = self.save(again, current2, prior2)
        data = torch.load(saved[2], weights_only=True)
        self.assertEqual(r.video_frames(data['video'].shape[2]), 651)
        self.assertTrue(torch.all(data['video'][:, :, :132] == .25))
        self.assertTrue(torch.all(data['video'][:, :, 132:] == .9))


if __name__ == '__main__':
    unittest.main(argv=[sys.argv[0]])
