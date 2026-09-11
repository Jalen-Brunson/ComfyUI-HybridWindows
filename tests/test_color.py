"""CPU checks for temporal color drift, source changes and brightness preservation."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()

spec = importlib.util.spec_from_file_location("video_color_test", ROOT / "color.py")
color = importlib.util.module_from_spec(spec)
spec.loader.exec_module(color)
torch.set_num_threads(2)


def luma(rgb):
    return rgb[..., 0] * .2126 + rgb[..., 1] * .7152 + rgb[..., 2] * .0722


def clip(cb_shift=0., cr_shift=0., saturation=1., count=320):
    rng = np.random.default_rng(17)
    y = rng.uniform(.25, .7, (24, 32)).astype(np.float32)
    cb = rng.uniform(-.04, .04, (24, 32)).astype(np.float32)
    cr = rng.uniform(-.03, .05, (24, 32)).astype(np.float32)
    cb = cb * np.asarray(saturation).reshape(-1, 1, 1) + np.asarray(cb_shift).reshape(-1, 1, 1)
    cr = cr * np.asarray(saturation).reshape(-1, 1, 1) + np.asarray(cr_shift).reshape(-1, 1, 1)
    r, b = y + 1.5748 * cr, y + 1.8556 * cb
    g = (y - .2126 * r - .0722 * b) / .7152
    rgb = np.stack(np.broadcast_arrays(r, g, b), axis=-1).astype(np.float32)
    return torch.from_numpy(np.broadcast_to(rgb, (count, 24, 32, 3)).copy())


def run(images, **kwargs):
    return color.VideoColorStabilize.execute(
        images, fps=8., reference_start_seconds=1., reference_end_seconds=3., **kwargs)[0]


def means(images):
    y = luma(images)
    return torch.stack(((images[..., 2] - y).mean((1, 2)) / 1.8556,
                        (images[..., 0] - y).mean((1, 2)) / 1.5748), dim=-1)


class ColorTests(unittest.TestCase):
    def test_bypass_is_exact_and_does_not_measure(self):
        images = clip()
        with patch.object(color, "color_stats", side_effect=AssertionError("measured bypass")):
            self.assertIs(run(images, enabled=False), images)
            self.assertIs(run(images, strength=0.), images)

    def test_constant_clip_stays_unchanged(self):
        images = clip()
        torch.testing.assert_close(run(images), images, rtol=0, atol=2e-7)

    def test_removes_known_slow_tint_and_saturation_drift(self):
        drift = np.clip((np.arange(320) - 40) / 180., 0., 1.)
        clean = clip()
        images = clip(.018 * drift, -.014 * drift, 1. - .09 * drift)
        snapshot = images.clone()
        result = run(images, strength=1.)
        original_error = (images[-32:] - clean[-32:]).abs().mean()
        corrected_error = (result[-32:] - clean[-32:]).abs().mean()
        self.assertLess(float(corrected_error), float(original_error) * .04)
        torch.testing.assert_close(result[8:24], images[8:24], rtol=0, atol=0)
        torch.testing.assert_close(luma(result), luma(images), rtol=0, atol=2e-7)
        torch.testing.assert_close(images, snapshot, rtol=0, atol=0)

    def test_source_changes_are_retained_relative_to_generated_anchor(self):
        drift = np.clip((np.arange(320) - 40) / 180., 0., 1.)
        source = clip(.01 * drift, .008 * drift)
        # The generated subject starts warmer than the original source.
        clean = clip(.01 * drift + .006, .008 * drift + .009)
        images = clip(.028 * drift + .006, -.006 * drift + .009)
        corrected = run(images, strength=1., source_frames=source)
        without_source = run(images, strength=1.)
        torch.testing.assert_close(means(corrected)[-32:], means(clean)[-32:], rtol=0, atol=2e-5)
        self.assertGreater(float((means(without_source)[-32:] - means(clean)[-32:]).abs().mean()), .005)

    def test_strong_blue_drift_is_not_stopped_by_old_hidden_limit(self):
        drift = np.clip((np.arange(320) - 40) / 180., 0., 1.)
        source = clip(-.004 * drift, .008 * drift)
        clean = clip(-.004 * drift + .006, .008 * drift + .009)
        images = clip(.041 * drift + .006, -.022 * drift + .009)
        corrected = run(images, strength=1., source_frames=source)
        limited = run(images, strength=1., source_frames=source, max_tint_shift=.025)
        torch.testing.assert_close(means(corrected)[-32:], means(clean)[-32:], rtol=0, atol=2e-5)
        self.assertGreater(float((means(limited)[-32:, 0] - means(clean)[-32:, 0]).mean()), .019)
        torch.testing.assert_close(luma(corrected), luma(images), rtol=0, atol=2e-7)

    def test_temporal_correction_has_no_step_at_a_chunk_boundary(self):
        drift = np.where(np.arange(320) < 160, 0., .018)
        images = clip(drift)
        corrected = run(images, strength=1.)
        applied = means(corrected) - means(images)
        # A color step in the input must not become a step in the grade curve.
        self.assertLess(float(torch.diff(applied[:, 0]).abs().max()), .001)
        self.assertLess(float(applied[-1, 0]), -.017)

    def test_gamut_limit_preserves_luma_for_saturated_pixels(self):
        rng = np.random.default_rng(11)
        rgb = rng.uniform(0., 1., (64, 64, 3)).astype(np.float32)
        rgb[0, :8] = [[0, 0, 0], [1, 1, 1], [1, 0, 0], [0, 1, 0],
                     [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]
        corrected = color.correct_frame(rgb, [0., 0.], 1.4, [.025, -.025])
        np.testing.assert_allclose(luma(corrected), luma(rgb), rtol=0, atol=2e-7)
        self.assertTrue(np.isfinite(corrected).all())
        self.assertGreaterEqual(corrected.min(), 0.)
        self.assertLessEqual(corrected.max(), 1.)

    def test_short_clip_and_different_source_resolution(self):
        images = clip(count=2)
        torch.testing.assert_close(run(images, source_frames=images[:, ::2, ::2]), images, rtol=0, atol=2e-7)

    def test_unchanged_zero_channel_does_not_block_other_color_changes(self):
        rgb = np.array([[[.4, .4, 0.]]], dtype=np.float32)
        corrected = color.correct_frame(rgb, [0., 0.], 1., [0., .01])
        self.assertGreater(float(corrected[0, 0, 0]), .41)
        self.assertEqual(corrected[0, 0, 2], 0.)
        np.testing.assert_allclose(luma(corrected), luma(rgb), rtol=0, atol=2e-7)

    def test_misaligned_source_fails_before_processing(self):
        images = clip()
        with self.assertRaisesRegex(ValueError, "same frame count"):
            run(images, source_frames=images[:1])
        with self.assertRaisesRegex(ValueError, "reference end"):
            color.VideoColorStabilize.execute(images, reference_start_seconds=8., reference_end_seconds=1.)

    def test_cancel_is_honored(self):
        with patch.object(color.comfy.model_management, "throw_exception_if_processing_interrupted",
                          side_effect=color.comfy.model_management.InterruptProcessingException):
            with self.assertRaises(color.comfy.model_management.InterruptProcessingException):
                run(clip())


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
