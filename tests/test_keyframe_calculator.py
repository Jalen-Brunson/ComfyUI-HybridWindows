"""Calculator results agree with the keyframe conditioning used by the sampler."""

import asyncio
import json
import unittest
from unittest.mock import patch

import test_keyframes as fixtures

import execution
from comfy_extras.nodes_preview_any import PreviewAny

Calculator = fixtures.chain.pack.H3HybridKeyframeCalculator


def calculate(total=175, window=56, overlap=5, mode="custom", **options):
    return Calculator.execute(total, window, overlap, {"placement": mode, **options})


class CalculatorTests(unittest.TestCase):
    def test_registered_with_native_schema(self):
        registered = asyncio.run(fixtures.chain.pack.HybridWindowsExtension().get_node_list())
        self.assertIn(Calculator, registered)
        schema = Calculator.define_schema()
        self.assertEqual(schema.node_id, "H3HybridKeyframeCalculator")
        self.assertIn("placement", Calculator.INPUT_TYPES()["required"])

    def test_example_and_all_modes_validate_and_execute_through_comfy(self):
        prompt = json.loads((fixtures.ROOT / "example_workflows" /
                             "H3 Hybrid Keyframe Calculator.api.json").read_text())
        mappings = {"H3HybridKeyframeCalculator": Calculator, "PreviewAny": PreviewAny}
        for mode, fields, expected in [
            ("custom", {"placement.frame_indices": "0, 51, 102, 153"}, "0, 51, 102, 153"),
            ("evenly spaced", {"placement.keyframe_count": 4}, "0, 58, 116, 174"),
            ("window starts", {}, "0, 51, 102, 119"),
        ]:
            with self.subTest(mode=mode), patch.dict(fixtures.chain.native_nodes.NODE_CLASS_MAPPINGS, mappings):
                prompt["1"]["inputs"] = {"total_frames": 175, "window_frames": 56,
                                          "overlap_frames": 5, "placement": mode, **fields}
                valid, _, _, errors = asyncio.run(execution.validate_prompt("calculator", prompt, None))
                self.assertTrue(valid, errors)
                inputs, missing, v3_data = execution.get_input_data(prompt["1"]["inputs"], Calculator, "1")
                self.assertFalse(missing)
                results = asyncio.run(execution._async_map_node_over_list(
                    "calculator", "1", Calculator, inputs, Calculator.FUNCTION, v3_data=v3_data))
                self.assertEqual(results[0][0], expected)
                preview = PreviewAny().main(results[0][2])
                self.assertIn("window 4: frames 119..174", preview["ui"]["text"][0])

    def test_report_reproduces_shifted_final_window_and_conditioning(self):
        result = calculate(frame_indices="0, 51, 102, 153")
        indices, total, report = result
        self.assertEqual(indices, "0, 51, 102, 153")
        self.assertEqual(total, 175)
        self.assertIn("window 4: frames 119..174 (56 frames), 39 frames shared with window 3", report)
        self.assertIn("Final window shifted back 34 frames", report)
        self.assertIn("use 209 total frames", report)
        placement = "frame 153 -> window 3 at its frame 51 and window 4 at its frame 34 (shared overlap)"
        self.assertIn(placement, report)

        payload, _ = fixtures.build(indices, {f"keyframe_{i}": fixtures.still(i / 4) for i in range(4)},
                                    width=32, height=32)
        source = fixtures.EmptyMiniMaxH3LatentAV.execute(32, 32, total)[0]
        groups = {f"positive_{i}": fixtures.chain.prompt(i) for i in range(4)}
        _, _, bound, _, _, sampler_report = fixtures.H3HybridWindows.execute(
            fixtures.chain.model(), 56, 5, groups, latent=source, keyframes=payload)
        self.assertIn(placement, sampler_report)
        self.assertEqual({k: [frame for frame, _ in values]
                          for k, values in fixtures.guides_by_window(bound).items()},
                         {0: [0, 51], 1: [0, 51], 2: [0, 51], 3: [34]})

    def test_window_starts_follow_actual_schedule(self):
        for total, window, overlap, expected in [
            (175, 56, 5, "0, 51, 102, 119"),
            (209, 56, 5, "0, 51, 102, 153"),
            (175, 39, 5, "0, 34, 68, 102, 136"),
            (175, 107, 39, "0, 68"),
            (22, 56, 5, "0"),
        ]:
            with self.subTest(total=total, window=window, overlap=overlap):
                indices, actual_total, _ = calculate(total, window, overlap, mode="window starts")
                self.assertEqual(indices, expected)
                self.assertEqual(actual_total, total)

    def test_regular_schedule_has_no_shift(self):
        _, _, report = calculate(209, frame_indices="0, 51, 102, 153")
        self.assertIn("window 4: frames 153..208 (56 frames), 5 frames shared with window 3", report)
        self.assertNotIn("shifted back", report)
        self.assertIn("window 4 at its frame 0", report)

    def test_even_spacing_includes_endpoints_and_describes_missing_context(self):
        indices, _, report = calculate(mode="evenly spaced", keyframe_count=4)
        self.assertEqual(indices, "0, 58, 116, 174")
        self.assertIn("no window sees both anchors for 0->58, 58->116, 116->174", report)
        indices, _, _ = calculate(mode="evenly spaced", keyframe_count=1)
        self.assertEqual(indices, "0")

    def test_rounding_matches_native_latent_and_window_alignment(self):
        indices, total, report = calculate(176, 57, 6, frame_indices="0, -1")
        source = fixtures.EmptyMiniMaxH3LatentAV.execute(32, 32, 176)[0]
        video = source["samples"].unbind()[0]
        self.assertEqual(total, fixtures.windows.frame_at(video.shape[2]))
        self.assertEqual((indices, total), ("0, 191", 192))
        self.assertIn("using 192/73/22", report)

    def test_custom_negatives_keep_image_order(self):
        indices, _, report = calculate(frame_indices="-1; 0\n-2")
        self.assertEqual(indices, "174, 0, 173")
        self.assertIn("keyframe 1: frame 174 -> window 4 at its frame 55", report)

    def test_short_run_uses_one_short_window(self):
        indices, _, report = calculate(5, mode="evenly spaced", keyframe_count=2)
        self.assertEqual(indices, "0, 4")
        self.assertIn("window 1: frames 0..4 (5 frames)", report)
        self.assertIn("every adjacent keyframe pair is visible together", report)

    def test_invalid_custom_frames_fail_before_sampling(self):
        for value, message in [("", "at least one"), ("0, 175", "outside"),
                               ("-176", "outside"), ("-1, 174", "listed twice"),
                               ("0, 1.5", "not a whole frame")]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                calculate(frame_indices=value)

    def test_no_duplicate_auto_frames_or_zero_stride(self):
        with self.assertRaisesRegex(ValueError, "unique frame"):
            calculate(5, mode="evenly spaced", keyframe_count=6)
        with self.assertRaisesRegex(ValueError, "Overlap must be smaller"):
            calculate(175, 56, 56, mode="window starts")


if __name__ == "__main__":
    unittest.main()
