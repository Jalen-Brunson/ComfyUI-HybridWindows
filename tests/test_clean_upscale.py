"""Check the clean-latent handoff against native re-noising and prefix preservation."""
import unittest
from unittest.mock import Mock, patch

import test_native_chain as native
from test_upscale import LearnedStandIn, MP, up
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV


class CleanUpscaleTests(unittest.TestCase):
    def warmup(self, split=6, source=None, accepted=0, start=0):
        model = native.model()
        source = source or EmptyMiniMaxH3LatentAV.execute(32, 32, 73)[0]
        groups = {f'positive_{i}': native.prompt(i) for i in range(2)}
        seq, joint, cond, _, prepared, _ = native.hybrid.H3HybridWindows.execute(
            model, 39, 5, groups, latent=source, accepted_prefix_frames=accepted,
            start_window=start, noise_mode='per_window')
        sigmas = native.comfy.samplers.calculate_sigmas(model.get_model_object('model_sampling'), 'simple', 8)
        head, tail = native.custom.SplitSigmas.execute(sigmas, split)
        warm = native.custom.SamplerCustomAdvanced.execute(
            native.custom.Noise_RandomNoise(123), native.custom.BasicGuider.execute(seq, cond)[0],
            native.custom.KSamplerSelect.execute('euler')[0], head, prepared)
        return model, joint, cond, warm, tail

    def upscale(self, joint, warm, **kwargs):
        with patch.dict(native.native_nodes.NODE_CLASS_MAPPINGS, {'MinimaxH3LatentUpscaler3D': LearnedStandIn}):
            return up.H3HybridCleanLatentUpscale.execute(joint, warm[0], warm[1], 'test', MP, **kwargs)

    def finish(self, model, latent, cond, tail):
        return native.custom.SamplerCustomAdvanced.execute(
            native.custom.Noise_EmptyNoise(), native.custom.BasicGuider.execute(model, cond)[0],
            native.custom.KSamplerSelect.execute('euler')[0], tail, latent)[0]

    def test_matches_native_clean_renoise_control_at_several_sigmas(self):
        for split in (4, 6, 7):
            model, joint, cond, warm, tail = self.warmup(split)
            original = up.joint_run(joint).stash
            upgraded, latent, _ = self.upscale(joint, warm)
            self.assertIs(up.joint_run(joint).stash, original)
            current = up.joint_run(upgraded).stash
            old_raw = native.comfy.utils.unpack_latents(original.raw_state, original.signature['shapes'])
            new_raw = native.comfy.utils.unpack_latents(current.raw_state, current.signature['shapes'])
            old_noise = native.comfy.utils.unpack_latents(original.noise, original.signature['shapes'])
            new_noise = native.comfy.utils.unpack_latents(current.noise, current.signature['shapes'])
            torch.testing.assert_close(new_raw[1], old_raw[1], rtol=0, atol=0)
            torch.testing.assert_close(new_noise[1], old_noise[1], rtol=0, atol=0)
            self.assertFalse(torch.equal(new_noise[0][..., ::2], new_noise[0][..., 1::2]))
            expected_clean = up.resize_video(warm[1]['samples'].unbind()[0], (4, 4)) + .03125
            injected = native.custom.AddNoise.execute(
                model, native.custom.Noise_RandomNoise(124), tail, {'samples': expected_clean})[0]
            reference_video = injected['samples'] / (1 - tail[0])
            torch.testing.assert_close(latent['samples'].unbind()[0], reference_video, rtol=1e-6, atol=1e-6)
            actual = self.finish(upgraded, latent, cond, tail)
            self.assertEqual(len(model.model.seen), 16)
            _, plain_joint, plain_cond, *_ = native.hybrid.H3HybridWindows.execute(
                native.model(), 39, 5, {f'positive_{i}': native.prompt(i) for i in range(2)})
            reference = self.finish(plain_joint, {'samples': NestedTensor([
                reference_video, warm[0]['samples'].unbind()[1]])}, plain_cond, tail)
            for got, expected in zip(actual['samples'].unbind(), reference['samples'].unbind()):
                torch.testing.assert_close(got, expected, rtol=2e-5, atol=2e-6)

    def test_resumed_high_resolution_prefix_is_exact(self):
        prior = EmptyMiniMaxH3LatentAV.execute(64, 64, 107)[0]
        generator = torch.Generator().manual_seed(43)
        prior['samples'] = NestedTensor([torch.randn(p.shape, generator=generator) for p in prior['samples'].unbind()])
        prior['h3_meta'] = {'start_chunk': 0, 'chunk_frames': 39, 'overlap_frames': 5}
        smaller = up.H3HybridWarmupPrior.execute(prior, 32, 32)[0]
        source = EmptyMiniMaxH3LatentAV.execute(32, 32, 73)[0]
        pv, pa = smaller['samples'].unbind()
        source['samples'].unbind()[0][:, :, :12] = pv[:, :, 10:22]
        source['samples'].unbind()[1][..., :65] = pa[..., 57:122]
        _, joint, cond, warm, tail = self.warmup(source=source, accepted=39, start=1)
        upgraded, latent, _ = self.upscale(joint, warm, prior=prior)
        result = self.finish(upgraded, latent, cond, tail)
        torch.testing.assert_close(result['samples'].unbind()[0][:, :, :12], prior['samples'].unbind()[0][:, :, 10:22], rtol=0, atol=0)
        torch.testing.assert_close(result['samples'].unbind()[1][..., :65], prior['samples'].unbind()[1][..., 57:122], rtol=0, atol=0)

    def test_source_masks_still_preserve_clean_source_and_audio(self):
        source = EmptyMiniMaxH3LatentAV.execute(32, 32, 73)[0]
        video, audio = source['samples'].unbind();video.fill_(.6);audio.fill_(.8)
        vm, am = torch.ones_like(video), torch.zeros_like(audio);vm[..., 0] = 0
        source['noise_mask'] = NestedTensor([vm, am])
        _, joint, cond, warm, tail = self.warmup(source=source)
        upgraded, latent, _ = self.upscale(joint, warm)
        result = self.finish(upgraded, latent, cond, tail)
        torch.testing.assert_close(result['samples'].unbind()[0][..., :2], torch.full_like(video[..., :1].repeat_interleave(2, -1).repeat_interleave(2, -2), .63125), rtol=0, atol=3e-7)
        torch.testing.assert_close(result['samples'].unbind()[1], audio, rtol=0, atol=3e-7)

    def test_vae_warmup_prior_keeps_audio_masks_and_timeline(self):
        prior = EmptyMiniMaxH3LatentAV.execute(64, 64, 39)[0]
        video, audio = prior["samples"].unbind()
        video.fill_(.3)
        prior["h3_meta"] = {"start_chunk": 2, "chunk_frames": 39, "overlap_frames": 5}
        prior["noise_mask"] = NestedTensor([torch.zeros_like(video), torch.ones_like(audio)])
        encoded = torch.full((*video.shape[:3], 2, 2), .8)
        vae = Mock()
        vae.decode.return_value = torch.full((1, 39, 64, 64, 3), .25)
        vae.encode.return_value = encoded
        result = up.H3HybridWarmupPrior.execute(prior, 32, 32, vae)[0]
        self.assertIs(result["samples"].unbind()[0], encoded)
        self.assertIs(result["samples"].unbind()[1], audio)
        self.assertEqual(result["h3_meta"], prior["h3_meta"])
        self.assertEqual(tuple(vae.encode.call_args.args[0].shape), (39, 32, 32, 3))
        self.assertTrue(torch.all(result["noise_mask"].unbind()[0] == 0))
        torch.testing.assert_close(video, torch.full_like(video, .3), rtol=0, atol=0)
        vae.reset_mock()
        self.assertIsNone(up.H3HybridWarmupPrior.execute(None, 32, 32, vae)[0])
        up.H3HybridWarmupPrior.execute(prior, 64, 64, vae)
        vae.decode.assert_not_called()
        vae.encode.assert_not_called()

    def test_vae_warmup_prior_rejects_changed_temporal_layout(self):
        prior = EmptyMiniMaxH3LatentAV.execute(64, 64, 39)[0]
        vae = Mock()
        vae.decode.return_value = torch.zeros(39, 64, 64, 3)
        vae.encode.return_value = torch.zeros(1, 24, 13, 2, 2)
        with self.assertRaisesRegex(ValueError, "temporal layout"):
            up.H3HybridWarmupPrior.execute(prior, 32, 32, vae)

    def test_wrong_denoised_shape_and_same_size(self):
        _, joint, _, warm, _ = self.warmup()
        wrong = {'samples': NestedTensor([warm[1]['samples'].unbind()[0][..., :1], warm[1]['samples'].unbind()[1]])}
        with self.assertRaisesRegex(ValueError, 'same warmup sampler'):
            self.upscale(joint, [warm[0], wrong])
        upgraded, latent, _ = up.H3HybridCleanLatentUpscale.execute(
            joint, warm[0], warm[1], 'unused', 32 * 32 / (1024 * 1024))
        self.assertIs(upgraded, joint);self.assertIs(latent, warm[0])


if __name__ == '__main__':
    unittest.main(argv=[__file__])
