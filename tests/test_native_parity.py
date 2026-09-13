"""The native two-sampler chain against the single-node sampler, same seeds.

The point of `noise_mode="per_window"` is that a graph can be moved from
MMH3HybridWindowSampler to the native chain without the output changing, so
this compares the two on one model with masks, an accepted prefix and a
resume shift. Needs MMH3Tools (set MMH3TOOLS_TEST_ROOT if it lives elsewhere).
"""

import importlib
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import test_native_chain as native  # bootstraps comfy in --cpu mode and loads the pack

sys.path[:0] = [os.environ.get("MMH3TOOLS_TEST_ROOT", str(ROOT.parent / "ComfyUI-MMH3Tools"))]
import torch
import comfy.nested_tensor
import comfy.samplers
import comfy.utils
import comfy_extras.nodes_custom_sampler as custom
import nodes as native_nodes
from comfy_extras.nodes_minimax_h3 import video_latent_t

try:
    from mmh3tools.nodes_looping_sampler import MMH3LoopingSampler
    native_nodes.NODE_CLASS_MAPPINGS["MMH3LoopingSampler"] = MMH3LoopingSampler
    single = importlib.import_module("hybrid_windows_test.sampler")
    MMH3_AVAILABLE = True
except Exception as error:  # pragma: no cover - depends on the sibling pack
    MMH3_AVAILABLE = False
    REASON = str(error)

hybrid, windows = native.hybrid, native.windows


def master(frames, keep_video=0):
    latent = native.EmptyMiniMaxH3LatentAV.execute(32, 32, frames)[0]
    video, audio = latent["samples"].unbind()
    video = video + torch.linspace(0, 1, video.shape[2]).reshape(1, 1, -1, 1, 1)
    audio = audio + .5
    out = {"samples": comfy.nested_tensor.NestedTensor([video, audio])}
    if keep_video:
        mask = windows.ones_mask(video, audio)
        mask[0][:, :, :keep_video] = 0
        out["noise_mask"] = comfy.nested_tensor.NestedTensor(mask)
    return out


@unittest.skipUnless(MMH3_AVAILABLE, "MMH3Tools not importable" if MMH3_AVAILABLE else globals().get("REASON", ""))
class ParityTests(unittest.TestCase):
    WINDOW, OVERLAP, SPLIT, STEPS, SEED = 39, 5, 6, 8, 123

    def both(self, frames, count, keep_video=0, accepted=0, start_window=0,
             window=39, overlap=5, split=6, overlap_pin="full", context_frames=5):
        source = master(frames, keep_video)
        conds = [native.prompt(i) for i in range(count)]
        cond_set = {"conds": conds, "prompts": [str(i) for i in range(count)]}

        one = native.model()
        sigmas = comfy.samplers.calculate_sigmas(
            one.get_model_object("model_sampling"), "simple", self.STEPS)
        reference = single.MMH3HybridWindowSampler.execute(
            one, custom.Noise_RandomNoise(self.SEED), custom.KSamplerSelect.execute("euler")[0],
            sigmas, cond_set, source, window, overlap, split, "cpu", "max",
            accepted_prefix_frames=accepted, start_window=start_window,
            overlap_pin=overlap_pin, context_frames=context_frames)[0]

        two = native.model()
        seq, joint, cond, _, prepared, _ = hybrid.H3HybridWindows.execute(
            two, window, overlap, cond_set=cond_set, latent=source,
            total_steps=self.STEPS, accepted_prefix_frames=accepted,
            start_window=start_window, noise_mode="per_window",
            overlap_pin=overlap_pin, context_frames=context_frames)
        high, low = custom.SplitSigmas.execute(sigmas, split)
        warm = custom.SamplerCustomAdvanced.execute(
            custom.Noise_RandomNoise(self.SEED), custom.BasicGuider.execute(seq, cond)[0],
            custom.KSamplerSelect.execute("euler")[0], high, prepared)[0]
        result = custom.SamplerCustomAdvanced.execute(
            custom.Noise_EmptyNoise(), custom.BasicGuider.execute(joint, cond)[0],
            custom.KSamplerSelect.execute("euler")[0], low, warm)[0]
        return one, two, reference, result

    def assert_same(self, reference, result, rtol=2e-6, atol=2e-6):
        for got, want in zip(result["samples"].unbind(), reference["samples"].unbind()):
            torch.testing.assert_close(got, want, rtol=rtol, atol=atol)

    def test_warmup_calls_are_identical(self):
        one, two, reference, result = self.both(73, 2)
        warm_one = [c for c in one.model.seen if len(c["sigmas"]) == self.SPLIT + 1]
        warm_two = [c for c in two.model.seen if len(c["sigmas"]) == self.SPLIT + 1]
        self.assertEqual(len(warm_one), len(warm_two))
        self.assertTrue(warm_one)
        for a, b in zip(warm_one, warm_two):
            torch.testing.assert_close(a["x"], b["x"], rtol=0, atol=0)
            torch.testing.assert_close(a["sigma"], b["sigma"], rtol=0, atol=0)
        self.assert_same(reference, result)

    def test_masked_source_matches(self):
        _, _, reference, result = self.both(73, 2, keep_video=4)
        self.assert_same(reference, result)

    def test_accepted_prefix_and_start_window_match(self):
        _, _, reference, result = self.both(73, 2, accepted=39, start_window=3)
        self.assert_same(reference, result)
        accepted_v = video_latent_t(39)
        video = result["samples"].unbind()[0]
        source_video = reference["samples"].unbind()[0]
        torch.testing.assert_close(video[:, :, :accepted_v], source_video[:, :, :accepted_v],
                                   rtol=0, atol=0)

    def test_partial_context_matches_native_masks_and_preserves_accepted_av(self):
        for frames, held in ((0, 0), (5, 2), (6, 3), (9, 3), (17, 5), (39, 12), (100, 12)):
            with self.subTest(context_frames=frames):
                one, two, reference, result = self.both(
                    107, 2, window=73, overlap=39, accepted=73, start_window=3,
                    overlap_pin="tail_custom", context_frames=frames)
                self.assert_same(reference, result)
                warm_one = [c for c in one.model.seen if len(c["sigmas"]) == self.SPLIT + 1]
                warm_two = [c for c in two.model.seen if len(c["sigmas"]) == self.SPLIT + 1]
                self.assertEqual(len(warm_one), self.SPLIT)
                for a, b in zip(warm_one, warm_two, strict=True):
                    torch.testing.assert_close(a["x"], b["x"], rtol=0, atol=0)
                    self.assertEqual(a["masks"].keys(), b["masks"].keys())
                    for key in a["masks"]:
                        torch.testing.assert_close(a["masks"][key], b["masks"][key], rtol=0, atol=0)
                if held:
                    mask = warm_one[0]["masks"]["denoise_mask"]
                    self.assertTrue(torch.all(mask[:, :, :12-held] == 1))
                    self.assertTrue(torch.all(mask[:, :, 12-held:12] == 0))
                else:
                    self.assertNotIn("denoise_mask", warm_one[0]["masks"])
                self.assertTrue(torch.all(warm_one[0]["masks"]["audio_denoise_mask"][..., :65] == 0))
                source = master(107)["samples"].unbind()
                for output in (reference, result):
                    video, audio = output["samples"].unbind()
                    nv = video_latent_t(73)
                    na = windows.audio_index_at(nv, video.shape[2], audio.shape[3])
                    torch.testing.assert_close(video[:, :, :nv], source[0][:, :, :nv], rtol=0, atol=0)
                    torch.testing.assert_close(audio[:, :, :, :na], source[1][:, :, :, :na], rtol=0, atol=0)

    def test_partial_context_matches_across_fresh_resumed_and_sliding_windows(self):
        for total, accepted in ((209, 0), (209, 73), (192, 73)):
            for split in (6, 8):
                with self.subTest(total=total, accepted=accepted, split=split):
                    _, _, reference, result = self.both(
                        total, 5, window=73, overlap=39, accepted=accepted, split=split,
                        start_window=2 if accepted else 0, overlap_pin="tail_custom", context_frames=5)
                    self.assert_same(reference, result)

    def test_partial_context_refuses_source_pins_outside_accepted_prefix(self):
        with self.assertRaisesRegex(ValueError, "full-frame regeneration"):
            self.both(107, 2, window=73, overlap=39, keep_video=25, accepted=73,
                      overlap_pin="tail_custom", context_frames=5)

    def test_single_node_feather_preserves_source_and_audio_masks(self):
        source = master(107)
        video, audio = source["samples"].unbind()
        mask = windows.ones_mask(video, audio)
        mask[0][:, :, 22] = 0  # Protect a whole DiT patch, not one cell of its 2x2 pooling.
        mask[1].zero_()
        source["noise_mask"] = comfy.nested_tensor.NestedTensor(mask)
        m = native.model()
        sigmas = comfy.samplers.calculate_sigmas(m.get_model_object("model_sampling"), "simple", 8)
        single.MMH3HybridWindowSampler.execute(
            m, custom.Noise_RandomNoise(self.SEED), custom.KSamplerSelect.execute("euler")[0],
            sigmas, {"conds": [native.prompt(0), native.prompt(1)]}, source, 73, 39, 8,
            feather_latents=4)
        second = next(c for c in m.model.seen if c["prompt"] == 1)
        vm = second["masks"]["denoise_mask"]
        torch.testing.assert_close(vm[0, 0, 12:16, 1, 1], torch.tensor([0., .5, .75, 1.]), rtol=0, atol=0)
        self.assertEqual(float(vm[0, 0, 12, 0, 0]), 0)
        self.assertTrue(torch.all(second["masks"]["audio_denoise_mask"] == 0))

    def test_three_windows_match(self):
        _, _, reference, result = self.both(107, 3)
        self.assert_same(reference, result)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
