"""Aligned guides through real native sampling, upscale, resume and standalone sampling."""

import importlib
import sys
import unittest
from unittest.mock import patch

import test_native_chain as native
import torch
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV, MiniMaxH3AddGuide, video_latent_t

guides = importlib.import_module("hybrid_windows_test.guides")
if (native.ROOT / "upscale.py").exists():
    from test_upscale import LearnedStandIn, MP, up
else:
    up = None


class VideoVAE:
    def __init__(self):
        self.seen = []

    def encode(self, image):
        self.seen.append((tuple(image.shape), round(float(image[0, 0, 0, 0]) * 255),
                          round(float(image[-1, 0, 0, 0]) * 255)))
        t = 1 if len(image) == 1 else video_latent_t(len(image))
        return torch.full((1, 24, t, image.shape[1] // 16, image.shape[2] // 16), float(image.mean()))


class AudioVAE:
    audio_sample_rate = 24000

    def __init__(self):
        self.seen = []

    def encode(self, waveform):
        self.seen.append((float(waveform[0, 0, 0]), waveform.shape[1]))
        return torch.zeros(1, 32, 2, round(waveform.shape[1] / self.audio_sample_rate * 40))


def frames(count=107, offset=0):
    return (torch.arange(offset, offset + count, dtype=torch.float32) / 255)[:, None, None, None].expand(-1, 32, 32, 3)


def soundtrack(count=107):
    return {"sample_rate": 24000, "waveform": torch.arange(count * 1000, dtype=torch.float32)[None, None, :]}


class GuideTests(unittest.TestCase):
    def chain(self, total=107, accepted=0, start=0, audio=False, video=True):
        m, vae, avae = native.model(), VideoVAE(), AudioVAE()
        source = EmptyMiniMaxH3LatentAV.execute(32, 32, total)[0]
        groups = {f"positive_{i}": native.prompt(i) for i in range(3)}
        seq, joint, cond, _, prepared, _ = native.hybrid.H3HybridWindows.execute(
            m, 39, 5, groups, latent=source, accepted_prefix_frames=accepted, start_window=start,
            guide_video=frames(total, 10) if video else None, guide_vae=vae,
            guide_audio=soundtrack(total) if audio else None, guide_audio_vae=avae)
        first = native.sample(seq, prepared, cond, end=6, leftover="enable")
        result = native.sample(joint, first, cond, start=6, add_noise="disable")
        return m, vae, avae, cond, source, result

    def test_windows_receive_their_own_clip_in_both_stages(self):
        m, vae, _, cond, _, _ = self.chain()
        self.assertEqual([(a, b) for _, a, b in vae.seen], [(10, 48), (44, 82), (78, 116)] * 2)
        self.assertEqual(len(m.model.seen), 24)
        for call in m.model.seen:
            kfs = call["payload"]["keyframes"]
            self.assertEqual(len(kfs), 1)
            self.assertEqual(kfs[0]["resolved_frame_index"], 0)
            self.assertEqual(tuple(kfs[0]["latent"].shape[2:]), (12, 2, 2))
            expected = (29 + 34 * int(call["prompt"])) / 255
            self.assertAlmostEqual(float(kfs[0]["latent"].mean()), expected, places=5)
        self.assertTrue(all("minimax_keyframes" not in metadata for _, metadata in cond))

    def test_last_window_slide_and_resume_do_not_double_offset(self):
        m, vae, _, _, source, result = self.chain(total=90, accepted=39, start=5)
        self.assertEqual([(a, b) for _, a, b in vae.seen], [(10, 48), (44, 82), (61, 99)] * 2)
        for got, want, length in zip(result["samples"].unbind(), source["samples"].unbind(), (12, 65)):
            dim = 2 if got.ndim == 5 else 3
            torch.testing.assert_close(got.narrow(dim, 0, length), want.narrow(dim, 0, length), rtol=0, atol=0)
        self.assertEqual(len(m.model.seen), 18)

    def test_audio_slices_and_audio_only_guides(self):
        for with_video in (True, False):
            m, _, avae, _, _, _ = self.chain(audio=True, video=with_video)
            self.assertEqual(avae.seen, [(0., 39000), (34000., 39000), (68000., 39000)] * 2)
            for call in m.model.seen:
                kf = call["payload"]["keyframes"][0]
                self.assertEqual(kf["audio_latent"].shape[-1], 65)
                self.assertEqual("latent" in kf, with_video)

    def test_cloned_joint_model_encodes_at_the_finishing_latent_size(self):
        m, vae = native.model(), VideoVAE()
        source = EmptyMiniMaxH3LatentAV.execute(32, 32, 73)[0]
        groups = {f"positive_{i}": native.prompt(i) for i in range(2)}
        seq, joint, cond, *_ = native.hybrid.H3HybridWindows.execute(
            m, 39, 5, groups, guide_video=frames(73), guide_vae=vae)
        first = native.sample(seq, source, cond, end=6, leftover="enable")
        video, audio = first["samples"].unbind()
        high = {"samples": NestedTensor([video.repeat_interleave(2, -2).repeat_interleave(2, -1), audio])}
        native.sample(joint.clone(), high, cond, start=6, add_noise="disable")
        self.assertEqual([shape[1:3] for shape, _, _ in vae.seen], [(32, 32)] * 2 + [(64, 64)] * 2)
        for call in m.model.seen:
            self.assertEqual(call["payload"]["keyframes"][0]["latent"].shape[-2:], call["shapes"][0][-2:])

    @unittest.skipIf(up is None, "Local hybrid upscaler extension is not installed")
    def test_both_upscalers_encode_again_at_actual_finishing_size(self):
        for node in (up.H3HybridLatentUpscale, up.H3HybridCleanLatentUpscale):
            m, vae = native.model(), VideoVAE()
            source = EmptyMiniMaxH3LatentAV.execute(32, 32, 73)[0]
            groups = {f"positive_{i}": native.prompt(i) for i in range(2)}
            seq, joint, cond, _, prepared, _ = native.hybrid.H3HybridWindows.execute(
                m, 39, 5, groups, latent=source, guide_video=frames(73), guide_vae=vae)
            sigmas = native.comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), "simple", 8)
            head, tail = native.custom.SplitSigmas.execute(sigmas, 6)
            warm = native.custom.SamplerCustomAdvanced.execute(
                native.custom.Noise_RandomNoise(123), native.custom.BasicGuider.execute(seq, cond)[0],
                native.custom.KSamplerSelect.execute("euler")[0], head, prepared)
            kwargs = {"denoised": warm[1]} if node is up.H3HybridCleanLatentUpscale else {}
            with patch.dict(native.native_nodes.NODE_CLASS_MAPPINGS, {"MinimaxH3LatentUpscaler3D": LearnedStandIn}):
                upgraded, latent, _ = node.execute(joint, warm[0], model_name="test", megapixels=MP, **kwargs)
            native.custom.SamplerCustomAdvanced.execute(
                native.custom.Noise_EmptyNoise(), native.custom.BasicGuider.execute(upgraded, cond)[0],
                native.custom.KSamplerSelect.execute("euler")[0], tail, latent)
            self.assertEqual([shape[1:3] for shape, _, _ in vae.seen], [(32, 32)] * 2 + [(64, 64)] * 2)
            for call in m.model.seen:
                self.assertEqual(call["payload"]["keyframes"][0]["latent"].shape[-2:], call["shapes"][0][-2:])

    def test_matches_explicit_native_add_guide_and_preserves_existing_still(self):
        source = EmptyMiniMaxH3LatentAV.execute(32, 32, 39)[0]
        original = native.image_prompt(first=True)
        explicit = MiniMaxH3AddGuide.execute(original, source, 0, vae=VideoVAE(), image=frames(39)).args[0]
        expected = native.sample(native.model(), source, explicit)
        m = native.model()
        seq, joint, cond, _, prepared, _ = native.hybrid.H3HybridWindows.execute(
            m, 39, 5, {"positive_0": original}, latent=source, guide_video=frames(39), guide_vae=VideoVAE())
        first = native.sample(seq, prepared, cond, end=6, leftover="enable")
        result = native.sample(joint, first, cond, start=6, add_noise="disable")
        for got, want in zip(result["samples"].unbind(), expected["samples"].unbind()):
            torch.testing.assert_close(got, want, rtol=2e-6, atol=1e-6)
        self.assertEqual(len(original[0][1]["minimax_keyframes"]), 1)
        self.assertTrue(all(len(call["payload"]["keyframes"]) == 2 for call in m.model.seen))

    def test_validation_and_failed_sampling_do_not_mutate_conditioning(self):
        with self.assertRaisesRegex(ValueError, "guide_vae"):
            guides.AlignedGuides(frames())
        with self.assertRaisesRegex(ValueError, "guide_audio_vae"):
            guides.AlignedGuides(audio=soundtrack())
        parts = EmptyMiniMaxH3LatentAV.execute(32, 32, 107)[0]["samples"].unbind()
        spans = native.windows.plan_windows(39, 5, 3).spans([p.shape for p in parts])
        with self.assertRaisesRegex(ValueError, "107 source frames"):
            guides.AlignedGuides(frames(73), VideoVAE()).encode(parts, spans)
        with self.assertRaisesRegex(ValueError, "shorter"):
            guides.AlignedGuides(audio=soundtrack(39), audio_vae=AudioVAE()).encode(parts, spans)
        m, vae = native.model(), VideoVAE()
        source = EmptyMiniMaxH3LatentAV.execute(32, 32, 39)[0]
        seq, _, cond, *_ = native.hybrid.H3HybridWindows.execute(
            m, 39, 5, {"positive_0": native.prompt()}, latent=source, guide_video=frames(39), guide_vae=vae)
        with patch.object(m.model, "apply_model", side_effect=RuntimeError("test interruption")):
            with self.assertRaisesRegex(RuntimeError, "test interruption"):
                native.sample(seq, source, cond, end=6, leftover="enable")
        native.sample(seq, source, cond, end=6, leftover="enable")
        self.assertTrue(all(len(call["payload"]["keyframes"]) == 1 for call in m.model.seen))
        self.assertNotIn("minimax_keyframes", cond[0][1])

    def test_standalone_matches_native_chain_with_guides(self):
        sys.path.insert(0, str(native.ROOT.parent / "ComfyUI-MMH3Tools"))
        from mmh3tools.nodes_looping_sampler import MMH3LoopingSampler
        native.native_nodes.NODE_CLASS_MAPPINGS["MMH3LoopingSampler"] = MMH3LoopingSampler
        single = importlib.import_module("hybrid_windows_test.sampler")
        cond_set = {"conds": [native.prompt(i) for i in range(3)]}
        source = EmptyMiniMaxH3LatentAV.execute(32, 32, 107)[0]
        a, b = native.model(), native.model()
        sigmas = native.comfy.samplers.calculate_sigmas(a.get_model_object("model_sampling"), "simple", 8)
        actual = single.MMH3HybridWindowSampler.execute(
            a, native.custom.Noise_RandomNoise(123), native.custom.KSamplerSelect.execute("euler")[0],
            sigmas, cond_set, source, 39, 5, 6, "cpu", "max", guide_video=frames(), guide_vae=VideoVAE())[0]
        seq, joint, cond, _, prepared, _ = native.hybrid.H3HybridWindows.execute(
            b, 39, 5, cond_set=cond_set, latent=source, noise_mode="per_window",
            guide_video=frames(), guide_vae=VideoVAE())
        first = native.sample(seq, prepared, cond, end=6, leftover="enable")
        expected = native.sample(joint, first, cond, start=6, add_noise="disable")
        for got, want in zip(actual["samples"].unbind(), expected["samples"].unbind()):
            torch.testing.assert_close(got, want, rtol=2e-6, atol=2e-6)
        for calls in (a.model.seen, b.model.seen):
            for call in calls:
                kf = call["payload"]["keyframes"][0]
                self.assertEqual(kf["resolved_frame_index"], 0)
                self.assertAlmostEqual(float(kf["latent"].mean()), (19 + int(call["prompt"]) * 34) / 255, places=5)
        self.assertTrue(all("minimax_keyframes" not in c[0][1] for c in cond_set["conds"]))


if __name__ == "__main__":
    unittest.main(argv=[__file__])
