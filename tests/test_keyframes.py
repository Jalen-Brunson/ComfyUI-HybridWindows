"""H3 Hybrid Keyframes: frame numbers of the whole run land in the right window at the right index."""

import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hybrid_native_chain_test", ROOT / "tests" / "test_native_chain.py")
chain = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = chain
spec.loader.exec_module(chain)
sys.argv = sys.argv[:1]  # the chain harness parked comfy's --cpu there; unittest must not see it

import torch
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV

keyframes = sys.modules["hybrid_windows_test.keyframes"]
windows = chain.windows
H3HybridKeyframes = keyframes.H3HybridKeyframes
H3HybridWindows = chain.hybrid.H3HybridWindows


class VAE:
    """Encodes a still to the 32x32 canvas' 2x2 latent, carrying the image's mean as its value."""

    def __init__(self):
        self.seen = []

    def encode(self, image):
        self.seen.append(tuple(image.shape))
        return torch.full((1, 24, 1, image.shape[1] // 16, image.shape[2] // 16), float(image.mean()))


def still(value, height=32, width=32, count=1):
    return torch.full((count, height, width, 3), value)


def build(frame_indices, images, vae=None, **options):
    return H3HybridKeyframes.execute(vae or VAE(), frame_indices, images=images, **options)


def guides_by_window(bound):
    out = {}
    for _, metadata in bound:
        window = metadata[windows.WINDOW_KEY]
        out[window] = [(g["resolved_frame_index"], round(float(g["latent"].mean()), 1))
                       for g in metadata.get("minimax_keyframes", [])]
    return out


class ParseTests(unittest.TestCase):
    def test_separators_and_negatives(self):
        self.assertEqual(keyframes.parse_indices("0, 15 32;64\n-1"), [0, 15, 32, 64, -1])
        self.assertEqual(keyframes.parse_indices(""), [])

    def test_bad_token_is_named(self):
        with self.assertRaisesRegex(ValueError, "'1.5' is not a whole frame number"):
            keyframes.parse_indices("0, 1.5")


class NodeTests(unittest.TestCase):
    def test_sockets_zip_with_numbers_in_socket_order(self):
        payload, report = build("0, 15, 32", {"keyframe_1": still(.9), "keyframe_0": still(.2, count=2)})
        self.assertEqual(payload["indices"], [0, 15, 32])
        self.assertEqual([name for name, _ in payload["stills"]], ["keyframe_0", "keyframe_0", "keyframe_1"])
        self.assertEqual([round(float(s.mean()), 1) for _, s in payload["stills"]], [.2, .2, .9])
        self.assertIn("3 keyframe(s)", report)

    def test_count_mismatch_names_both_counts(self):
        with self.assertRaisesRegex(ValueError, "2 image\\(s\\) but 3 frame number\\(s\\)"):
            build("0, 15, 32", {"keyframe_0": still(.2), "keyframe_1": still(.9)})

    def test_no_images(self):
        with self.assertRaisesRegex(ValueError, "at least one image"):
            build("0", {})

    def test_duplicate_frame_and_half_canvas(self):
        with self.assertRaisesRegex(ValueError, "frame 15 is listed twice"):
            build("15, 15", {"keyframe_0": still(.2), "keyframe_1": still(.9)})
        with self.assertRaisesRegex(ValueError, "both width and height"):
            build("0", {"keyframe_0": still(.2)}, width=32)


class WindowPlacementTests(unittest.TestCase):
    """Three windows of 39 frames, overlap 5 (stride 34): frames 0-38, 34-72, 68-106; 107 in all."""

    def groups(self, first=False):
        return {"positive_0": chain.image_prompt(0, first=first), "positive_1": chain.image_prompt(1),
                "positive_2": chain.image_prompt(2)}

    def test_global_frames_become_window_indices(self):
        payload, _ = build("0, 36, 49, 88, -1",
                           {f"keyframe_{i}": still(v) for i, v in enumerate((.1, .2, .3, .4, .5))},
                           width=32, height=32)
        _, _, bound, total, _, report = H3HybridWindows.execute(chain.model(), 39, 5, self.groups(), keyframes=payload)
        self.assertEqual(total, 107)
        self.assertEqual(guides_by_window(bound), {
            0: [(0, .1), (36, .2)],
            1: [(2, .2), (15, .3)],
            2: [(20, .4), (38, .5)],
        })
        self.assertIn("frame 36 -> window 1 at its frame 36 and window 2 at its frame 2 (shared overlap)", report)
        self.assertIn("frame 106 -> window 3 at its frame 38", report)

    def test_shared_still_is_encoded_once_per_keyframe(self):
        vae = VAE()
        payload, _ = build("36", {"keyframe_0": still(.2)}, vae=vae, width=32, height=32)
        H3HybridWindows.execute(chain.model(), 39, 5, self.groups(), keyframes=payload)
        self.assertEqual(len(vae.seen), 1)

    def test_still_is_fitted_to_the_canvas(self):
        vae = VAE()
        payload, _ = build("0", {"keyframe_0": still(.2, height=64, width=128)}, vae=vae, width=32, height=32)
        _, _, _, _, _, report = H3HybridWindows.execute(chain.model(), 39, 5, self.groups(), keyframes=payload)
        self.assertEqual(vae.seen, [(1, 32, 32, 3)])
        self.assertIn("fitted by crop from 128x64", report)

    def test_replaces_the_encoders_guide_at_the_same_frame(self):
        payload, _ = build("0", {"keyframe_0": still(.4)})
        _, _, bound, _, _, report = H3HybridWindows.execute(chain.model(), 39, 5, self.groups(first=True), keyframes=payload)
        self.assertEqual(guides_by_window(bound)[0], [(0, .4)])
        self.assertIn("replaces the encoder's guide there", report)
        self.assertIn("size from the encoders' own guide", report)

    def test_canvas_from_master_latent(self):
        payload, _ = build("50", {"keyframe_0": still(.4)})
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, 107)[0]
        _, _, bound, _, _, report = H3HybridWindows.execute(chain.model(), 39, 5, self.groups(), latent=latent, keyframes=payload)
        self.assertEqual(guides_by_window(bound)[1], [(16, .4)])
        self.assertIn("size from the master latent", report)

    def test_canvas_unknown_says_what_to_connect(self):
        payload, _ = build("0", {"keyframe_0": still(.4)})
        with self.assertRaisesRegex(ValueError, "Set its width and height"):
            H3HybridWindows.execute(chain.model(), 39, 5, self.groups(), keyframes=payload)

    def test_frame_outside_the_run(self):
        payload, _ = build("107", {"keyframe_0": still(.4)}, width=32, height=32)
        with self.assertRaisesRegex(ValueError, "frame 107 is outside this run's frames 0..106"):
            H3HybridWindows.execute(chain.model(), 39, 5, self.groups(), keyframes=payload)
        payload, _ = build("-1, 106", {"keyframe_0": still(.4), "keyframe_1": still(.5)}, width=32, height=32)
        with self.assertRaisesRegex(ValueError, "both name frame 106"):
            H3HybridWindows.execute(chain.model(), 39, 5, self.groups(), keyframes=payload)

    def test_accepted_prefix_is_reported(self):
        payload, _ = build("3, 60", {"keyframe_0": still(.4), "keyframe_1": still(.5)})
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, 107)[0]
        _, _, bound, _, _, report = H3HybridWindows.execute(
            chain.model(), 39, 5, self.groups(), latent=latent, accepted_prefix_frames=39, keyframes=payload)
        self.assertIn("frame 3 -> window 1 at its frame 3, inside the accepted prefix so it changes nothing", report)
        self.assertEqual(guides_by_window(bound)[1], [(26, .5)])

    def test_both_stages_see_each_window_only_its_own_guides(self):
        payload, _ = build("15, 49, 88", {f"keyframe_{i}": still(v) for i, v in enumerate((.1, .4, .9))},
                           width=32, height=32)
        m = chain.model()
        sequential, joint, cond, total, *_ = H3HybridWindows.execute(m, 39, 5, self.groups(), keyframes=payload)
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, total)[0]
        first = chain.sample(sequential, latent, cond, end=6, leftover="enable")
        chain.sample(joint, first, cond, start=6, add_noise="disable")
        self.assertEqual(len(m.model.seen), 24)
        seen = {}
        for i, call in enumerate(m.model.seen):
            stage = "sequential" if i < 18 else "joint"
            guides = tuple((g["resolved_frame_index"], round(float(g["latent"].mean()), 1))
                           for g in call["payload"].get("keyframes", []))
            seen.setdefault((stage, int(call["prompt"])), set()).add(guides)
        expected = {0: ((15, .1),), 1: ((15, .4),), 2: ((20, .9),)}
        for stage in ("sequential", "joint"):
            for window, guides in expected.items():
                self.assertEqual(seen[(stage, window)], {guides}, (stage, window))


if __name__ == "__main__":
    unittest.main()
