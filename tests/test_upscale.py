"""Exercise the real native sampler handoff, including resumed source pins."""

import importlib
import unittest
from unittest.mock import patch

import test_native_chain as native
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV, video_latent_t

up = importlib.import_module('hybrid_windows_test.upscale')
MP = 64 * 64 / (1024 * 1024)


class LearnedStandIn:
    unloads = []

    @classmethod
    def execute(cls, latent, model_name, mode, align, chunking, unload, device, precision):
        cls.unloads.append(unload)
        video = latent['samples']
        size = up.target_size(video, mode['megapixels'], align)
        return ({'samples': up.resize_video(video, size) + .03125},)


class UpscaleTests(unittest.TestCase):
    def warmup(self, accepted=0, start=0, mask=None, source=None):
        model = native.model()
        latent = source or EmptyMiniMaxH3LatentAV.execute(32, 32, 73)[0]
        groups = {'positive_0': native.prompt(0), 'positive_1': native.prompt(1)}
        seq, joint, cond, _, prepared, _ = native.hybrid.H3HybridWindows.execute(
            model, 39, 5, groups, latent=latent, accepted_prefix_frames=accepted,
            start_window=start, noise_mode='per_window')
        if mask is not None:
            raise AssertionError('Attach source masks to the input LATENT.')
        first = native.sample(seq, prepared, cond, end=6, leftover='enable')
        return model, joint, cond, first

    def upscale(self, joint, first, **kwargs):
        with patch.dict(native.native_nodes.NODE_CLASS_MAPPINGS, {'MinimaxH3LatentUpscaler3D': LearnedStandIn}):
            return up.H3HybridLatentUpscale.execute(joint, first, 'test', MP, **kwargs)

    def test_six_low_steps_then_two_high_steps_and_raw_audio_continuation(self):
        model, joint, cond, first = self.warmup()
        original = up.joint_run(joint)
        old_stash = original.stash
        before = len(model.model.seen)
        self.assertEqual(before, 12)
        upgraded, latent, report = self.upscale(joint, first)
        current = up.joint_run(upgraded)
        self.assertIs(up.joint_run(joint).stash, old_stash)
        self.assertEqual(current.stash.sigma_end, old_stash.sigma_end)
        old_shapes = old_stash.signature['shapes']
        new_shapes = current.stash.signature['shapes']
        old_raw = native.comfy.utils.unpack_latents(old_stash.raw_state, old_shapes)
        new_raw = native.comfy.utils.unpack_latents(current.stash.raw_state, new_shapes)
        sigma = current.stash.sigma_end
        expected_video = (up.resize_video(first['samples'].unbind()[0], (4, 4)) + .03125) * (1 - sigma)
        torch.testing.assert_close(new_raw[0], expected_video, rtol=0, atol=0)
        torch.testing.assert_close(new_raw[1], old_raw[1], rtol=0, atol=0)
        old_noise = native.comfy.utils.unpack_latents(old_stash.noise, old_shapes)
        new_noise = native.comfy.utils.unpack_latents(current.stash.noise, new_shapes)
        torch.testing.assert_close(new_noise[0], up.resize_video(old_noise[0], (4, 4)), rtol=0, atol=0)
        torch.testing.assert_close(new_noise[1], old_noise[1], rtol=0, atol=0)
        result = native.sample(upgraded, latent, cond, start=6, add_noise='disable')
        self.assertEqual(len(model.model.seen), 16)
        self.assertTrue(all(tuple(c['shapes'][0][-2:]) == (2, 2) for c in model.model.seen[:before]))
        self.assertTrue(all(tuple(c['shapes'][0][-2:]) == (4, 4) for c in model.model.seen[before:]))
        self.assertEqual(result['samples'].unbind()[1].shape, first['samples'].unbind()[1].shape)
        self.assertTrue(torch.isfinite(result['samples'].unbind()[0]).all())

    def test_resume_restores_original_high_resolution_prefix_and_audio(self):
        prior = EmptyMiniMaxH3LatentAV.execute(64, 64, 107)[0]
        generator = torch.Generator().manual_seed(43)
        prior['samples'] = NestedTensor([torch.randn(p.shape, generator=generator) for p in prior['samples'].unbind()])
        prior['h3_meta'] = {'start_chunk': 0, 'chunk_frames': 39, 'overlap_frames': 5}
        low_prior = up.H3HybridWarmupPrior.execute(prior, 32, 32)[0]
        source = EmptyMiniMaxH3LatentAV.execute(32, 32, 73)[0]
        video, audio = source['samples'].unbind()
        pv, pa = low_prior['samples'].unbind()
        video[:, :, :12] = pv[:, :, 10:22]
        audio[..., :65] = pa[..., 57:122]
        _, joint, cond, first = self.warmup(39, 1, source=source)
        upgraded, latent, _ = self.upscale(joint, first, prior=prior)
        result = native.sample(upgraded, latent, cond, start=6, add_noise='disable')
        rv, ra = result['samples'].unbind()
        hv, ha = prior['samples'].unbind()
        torch.testing.assert_close(rv[:, :, :12], hv[:, :, 10:22], rtol=0, atol=0)
        torch.testing.assert_close(ra[..., :65], ha[..., 57:122], rtol=0, atol=0)
        self.assertEqual(low_prior['h3_meta'], prior['h3_meta'])
        self.assertEqual(hv.shape[-1], 4)
        self.assertEqual(pv.shape[-1], 2)

    def test_inpaint_mask_preserves_upscaled_clean_source_and_audio(self):
        source = EmptyMiniMaxH3LatentAV.execute(32, 32, 73)[0]
        video, audio = source['samples'].unbind()
        video.fill_(.6)
        audio.fill_(.8)
        vm, am = torch.ones_like(video), torch.zeros_like(audio)
        vm[..., 0] = 0
        source['noise_mask'] = NestedTensor([vm, am])
        _, joint, cond, first = self.warmup(source=source)
        upgraded, latent, _ = self.upscale(joint, first)
        result = native.sample(upgraded, latent, cond, start=6, add_noise='disable')
        rv, ra = result['samples'].unbind()
        torch.testing.assert_close(rv[..., :2], torch.full_like(rv[..., :2], .63125), rtol=0, atol=3e-7)
        torch.testing.assert_close(ra, audio, rtol=0, atol=3e-7)

    def test_stale_input_and_incompatible_prior_fail_before_upscaling(self):
        _, joint, _, first = self.warmup()
        stale = {**first, 'samples': first['samples'] + .25}
        with self.assertRaisesRegex(ValueError, 'output slot 0'):
            self.upscale(joint, stale)
        _, joint, _, first = self.warmup(39)
        with self.assertRaisesRegex(ValueError, 'original saved master'):
            self.upscale(joint, first)
        prior = EmptyMiniMaxH3LatentAV.execute(32, 32, 73)[0]
        with self.assertRaisesRegex(ValueError, 'Saved master is'):
            self.upscale(joint, first, prior=prior)

    def test_same_resolution_is_the_existing_handoff(self):
        _, joint, cond, first = self.warmup()
        upgraded, latent, _ = up.H3HybridLatentUpscale.execute(joint, first, 'unused', 32 * 32 / (1024 * 1024))
        self.assertIs(upgraded, joint)
        self.assertIs(latent, first)
        native.sample(upgraded, latent, cond, start=6, add_noise='disable')

    def test_keep_loaded_toggle_drives_the_companion_unload(self):
        _, joint, _, first = self.warmup()
        LearnedStandIn.unloads.clear()
        self.upscale(joint, first)
        self.assertEqual(LearnedStandIn.unloads, [True])
        _, joint, _, first = self.warmup()
        LearnedStandIn.unloads.clear()
        self.upscale(joint, first, keep_upscaler_loaded=True)
        self.assertEqual(LearnedStandIn.unloads, [False])

    def test_all_accepted_and_fresh_lazy_prior(self):
        model = native.model()
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, 39)[0]
        seq, joint, cond, _, prepared, _ = native.hybrid.H3HybridWindows.execute(
            model, 39, 5, {'positive_0': native.prompt()}, latent=latent, accepted_prefix_frames=39)
        first = native.sample(seq, prepared, cond, end=6, leftover='enable')
        prior = EmptyMiniMaxH3LatentAV.execute(64, 64, 39)[0]
        prior['samples'].unbind()[0].fill_(.7)
        upgraded, latent, _ = self.upscale(joint, first, prior=prior)
        result = native.sample(upgraded, latent, cond, start=6, add_noise='disable')
        torch.testing.assert_close(result['samples'].unbind()[0], prior['samples'].unbind()[0], rtol=0, atol=0)
        self.assertEqual(model.model.seen, [])
        _, joint, _, _ = self.warmup()
        self.assertEqual(up.H3HybridLatentUpscale.check_lazy_status(joint), [])
        self.assertIsNone(up.H3HybridWarmupPrior.execute(None, 32, 32)[0])


if __name__ == '__main__':
    unittest.main(argv=[__file__])
