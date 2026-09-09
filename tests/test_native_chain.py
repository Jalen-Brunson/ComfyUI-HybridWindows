"""Exercise stock KSampler Advanced/CFGGuider/Euler with a tiny CPU H3 stand-in."""

import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parents[1]))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.conds
import comfy.latent_formats
import comfy.model_base
import comfy.model_patcher
import comfy.model_sampling
import comfy.utils
import comfy.samplers
import nodes as native_nodes
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV, MiniMaxH3ImageToVideo
from comfy.ldm.minimax.model import FinalLayer, MiniMaxH3Model
import comfy.ops

spec = importlib.util.spec_from_file_location("hybrid_windows_test", ROOT / "__init__.py",
                                            submodule_search_locations=[str(ROOT)])
pack = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = pack
spec.loader.exec_module(pack)
hybrid = sys.modules[spec.name + ".nodes"]
windows = sys.modules[spec.name + ".windows"]
control = sys.modules[spec.name + ".control"]
torch.set_num_threads(2)


class Sampling(comfy.model_sampling.ModelSamplingAV, comfy.model_sampling.CONST):
    pass


class ToyH3(comfy.model_base.MiniMaxH3):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self.diffusion_model = torch.nn.Linear(1, 1)
        self.diffusion_model.dtype = torch.float32
        self.diffusion_model.patch_size = (1, 2, 2)
        self.diffusion_model.preprocess_text_embeds = lambda x: x
        self.latent_format = comfy.latent_formats.MiniMaxH3AV()
        self.model_sampling = Sampling()
        self.model_sampling.set_parameters(shift=12, audio_shift=3)
        self.latent_shapes = None
        self.concat_keys = ()
        self.manual_cast_dtype = None
        self.model_config = SimpleNamespace(memory_usage_factor=1)
        self.seen = []

    def memory_required(self, input_shape, cond_shapes={}):
        return 0

    def apply_model(self, x, t, c_crossattn=None, transformer_options={}, **kwargs):
        shapes = kwargs["latent_shapes"]
        self.seen.append({"x": x.clone(), "sigma": t.clone(), "shapes": shapes,
                          "prompt": float(c_crossattn.mean()),
                          "offset": transformer_options.get(windows.OFFSET_KEY, 0),
                          "payload": kwargs["minimax_payload"],
                          "masks": {k: kwargs[k].clone() for k in ("denoise_mask", "audio_denoise_mask") if k in kwargs},
                          "sigmas": transformer_options["sample_sigmas"].clone()})
        # Deliberately context-dependent: independent windows cannot equal a
        # whole-timeline forward, making incorrect fusion/handoff observable.
        guide_rows = MiniMaxH3Model._cond_video_rows(self.diffusion_model, kwargs["minimax_payload"], x.device)
        guide_signal = 0 if guide_rows is None else .05*guide_rows.mean()
        return .27*x.tanh() + .13*x.mean() + .31*t.reshape(-1, 1, 1) + .02*c_crossattn.mean() + guide_signal


def model():
    result = comfy.model_patcher.ModelPatcher(ToyH3(), torch.device("cpu"), torch.device("cpu"))
    result.model.current_patcher = result
    return result


def prompt(index=0):
    return [[torch.full((1, 2, 3), float(index)), {}]]


def image_prompt(index=0, first=False, last=False):
    class Clip:
        def tokenize(self, text, images):
            return float(text)

        def encode_from_tokens_scheduled(self, tokens):
            return prompt(tokens)

    class VAE:
        def encode(self, image):
            return torch.full((1, 24, 1, 2, 2), float(image.mean()))

    return MiniMaxH3ImageToVideo.execute(
        Clip(), VAE(), str(index), 32, 32, 39,
        first_frame=torch.full((1, 32, 32, 3), .2) if first else None,
        last_frame=torch.full((1, 32, 32, 3), .8) if last else None)[0]


def sample(m, latent, positive, start=0, end=8, add_noise="enable", leftover="disable"):
    with patch("latent_preview.prepare_callback", return_value=None):
        return native_nodes.KSamplerAdvanced().sample(
            m, add_noise, 123, 8, 1.0, "euler", "simple", positive, positive,
            latent, start, end, leftover)[0]


class NativeChainTests(unittest.TestCase):
    def test_fl2va_boundary_guides_are_scoped_in_both_stages(self):
        groups = {"positive_0": image_prompt(0, first=True), "positive_1": image_prompt(1),
                  "positive_2": image_prompt(2, last=True)}
        m = model()
        sequential, joint, cond, total = hybrid.H3HybridWindows.execute(m, 39, 5, groups)
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, total)[0]
        first = sample(sequential, latent, cond, end=6, leftover="enable")
        sample(joint, first, cond, start=6, add_noise="disable")
        self.assertEqual(len(m.model.seen), 24)
        for call in m.model.seen:
            index = int(call["prompt"])
            payload = call["payload"]
            guides = payload.get("keyframes", [])
            self.assertEqual([g["resolved_frame_index"] for g in guides], {0: [0], 1: [], 2: [38]}[index])
            if guides:
                expected = groups[f"positive_{index}"][0][1]["minimax_keyframes"][0]["latent"]
                torch.testing.assert_close(guides[0]["latent"], expected, rtol=0, atol=0)
                torch.testing.assert_close(payload["cond_video_latents"][0], expected, rtol=0, atol=0)
        # Native metadata is reusable; no global-offset rewrite or carry guide is added.
        self.assertEqual(groups["positive_2"][0][1]["minimax_keyframes"][0]["resolved_frame_index"], 38)
        self.assertNotIn("minimax_keyframes", groups["positive_1"][0][1])

    def test_fl2va_single_window_preserves_native_guide_effect(self):
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, 39)[0]
        positive = image_prompt(first=True, last=True)
        reference = sample(model(), latent, positive)
        unguided = sample(model(), latent, prompt())
        self.assertFalse(torch.allclose(reference["samples"].unbind()[0], unguided["samples"].unbind()[0]))
        for split in range(1, 8):
            sequential, joint, cond, _ = hybrid.H3HybridWindows.execute(model(), 39, 5, {"positive_0": positive})
            first = sample(sequential, latent, cond, end=split, leftover="enable")
            result = sample(joint, first, cond, start=split, add_noise="disable")
            for actual, expected in zip(result["samples"].unbind(), reference["samples"].unbind()):
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)

    def test_single_window_matches_native_for_every_split(self):
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, 39)[0]
        original = model()
        reference = sample(original, latent, prompt())
        for split in range(1, 8):
            m = model()
            sequential, joint, cond, total = hybrid.H3HybridWindows.execute(
                m, 39, 5, {"positive_0": prompt()})
            first = sample(sequential, latent, cond, end=split, leftover="enable")
            result = sample(joint, first, cond, start=split, add_noise="disable")
            for actual, expected in zip(result["samples"].unbind(), reference["samples"].unbind()):
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)
            for observed, expected in zip(m.model.seen, original.model.seen, strict=True):
                torch.testing.assert_close(observed["x"], expected["x"], rtol=2e-6, atol=1e-6)
            self.assertNotIn("noise_mask", first)
            self.assertEqual(set(first), {"samples"})

    def test_full_sequential_split_finishes_and_second_sampler_is_passthrough(self):
        for count in (1, 3):
            m = model()
            sequential, joint, cond, total = hybrid.H3HybridWindows.execute(
                m, 39, 5, {f"positive_{i}": prompt(i) for i in range(count)}, total_steps=8)
            latent = EmptyMiniMaxH3LatentAV.execute(32, 32, total)[0]
            first = sample(sequential, latent, cond, end=8, leftover="enable")
            result = sample(joint, first, cond, start=8, add_noise="disable")
            self.assertEqual(len(m.model.seen), count * 8)
            for actual, expected in zip(result["samples"].unbind(), first["samples"].unbind()):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            if count == 1:
                reference = sample(model(), latent, prompt())
                for actual, expected in zip(result["samples"].unbind(), reference["samples"].unbind()):
                    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-6)

    def test_three_windows_pin_then_release_and_route_prompts(self):
        m = model()
        sequential, joint, cond, total = hybrid.H3HybridWindows.execute(
            m, 39, 5, {f"positive_{i}": prompt(i) for i in range(3)})
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, total)[0]
        original_parts = [x.clone() for x in latent["samples"].unbind()]
        first = sample(sequential, latent, cond, end=6, leftover="enable")
        result = sample(joint, first, cond, start=6, add_noise="disable")
        seen = m.model.seen
        self.assertEqual(len(seen), 3*8)
        self.assertEqual([s["prompt"] for s in seen], [0]*6 + [1]*6 + [2]*6 + [0, 1, 2]*2)
        self.assertEqual([s["offset"] for s in seen], [0]*6 + [34]*6 + [68]*6 + [0, 34, 68]*2)
        for s in seen[:6] + seen[18:]:
            self.assertEqual(s["masks"], {})
        for s in seen[6:18]:
            self.assertEqual(float(s["masks"]["denoise_mask"][:, :, :2].max()), 0)
            self.assertEqual(float(s["masks"]["denoise_mask"][:, :, 2:].min()), 1)
        for before, after in zip(original_parts, latent["samples"].unbind()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        for p in result["samples"].unbind():
            self.assertTrue(torch.isfinite(p).all())
        self.assertEqual(m.model_options, {"transformer_options": {}})

    def test_bad_chain_settings_fail_before_model_evaluation(self):
        m = model()
        sequential, joint, cond, total = hybrid.H3HybridWindows.execute(m, 39, 5, {"positive_0": prompt()})
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, total)[0]
        with self.assertRaisesRegex(ValueError, "leftover_noise"):
            sample(sequential, latent, cond, end=6)
        with self.assertRaisesRegex(ValueError, "Disable add_noise"):
            sample(joint, latent, cond, start=6)
        self.assertEqual(m.model.seen, [])

    def test_prompt_socket_numbers_define_order(self):
        m = model()
        _, _, bound, total = hybrid.H3HybridWindows.execute(
            m, 243, 39, {"positive_2": prompt(2), "positive_0": prompt(0), "positive_1": prompt(1)})
        self.assertEqual([float(c[0].mean()) for c in bound], [0, 1, 2])
        self.assertEqual([c[1][windows.WINDOW_KEY] for c in bound], [0, 1, 2])
        self.assertEqual(total, 651)

    def test_native_pdd_head_intervals_survive_schedule_split(self):
        # Execute core's actual widened PDD head path with tiny, known weights.
        layer = FinalLayer(4, 4, 2, 2, 1e-6, operations=comfy.ops.disable_weight_init)
        class Modulation(torch.nn.Module):
            def forward(self, t):
                return torch.zeros(1, 4), torch.zeros(1, 4)
        layer.adaln_proj = Modulation()
        layer.norm = torch.nn.Identity()
        for head in (layer.video_out, layer.audio_out):
            head.weight = torch.nn.Parameter(torch.arange(32*2*4).reshape(64, 4).float()/1000)
            head.bias = torch.nn.Parameter(torch.arange(64).float()/100)
        sigmas = comfy.samplers.calculate_sigmas(model().model.model_sampling, "simple", 8)
        x = torch.arange(16).reshape(4, 4).float()/10
        def forward(s, schedule):
            return layer(x, torch.zeros(1, 4), (0, 2, 0), (2, 4, 0), s, schedule, (12, 3))
        for split in range(1, 8):
            for i in range(8):
                schedule = sigmas[:split+1] if i < split else sigmas[split:]
                for actual, expected in zip(forward(sigmas[i], schedule), forward(sigmas[i], sigmas)):
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_joint_fusion_is_weighted_prediction_before_one_update(self):
        plan = windows.plan_windows(39, 5, 3)
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, plan.total_frames)[0]
        packed, shapes = comfy.utils.pack_latents(latent["samples"].unbind())
        cond = [{windows.WINDOW_KEY: i, "model_conds": {"latent_shapes": comfy.conds.CONDConstant(shapes)}}
                for i in range(3)]
        def predict(model, groups, sub_x, sigma, options):
            i = groups[0][0][windows.WINDOW_KEY]
            return [torch.full_like(sub_x, 10.0*(i+1)), torch.zeros_like(sub_x)]
        out = windows.JointWindows(plan).execute(predict, None, [cond, None], packed, torch.tensor([.8]), {})
        video, audio = comfy.utils.unpack_latents(out[0], shapes)
        self.assertAlmostEqual(float(video[0, 0, 10, 0, 0]), (10*2+20*1)/3, places=5)
        self.assertAlmostEqual(float(video[0, 0, 11, 0, 0]), (10*1+20*2)/3, places=5)
        self.assertEqual(float(video[0, 0, 0, 0, 0]), 10)
        self.assertEqual(float(video[0, 0, -1, 0, 0]), 30)
        self.assertEqual(audio.shape[2], 2)
        self.assertTrue(torch.isfinite(out[0]).all())

    def test_cancel_restores_execution_local_options(self):
        m = model()
        sequential, _, cond, total = hybrid.H3HybridWindows.execute(m, 39, 5, {"positive_0": prompt()})
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, total)[0]
        with patch.object(m.model, "apply_model", side_effect=RuntimeError("cancelled")):
            with self.assertRaisesRegex(RuntimeError, "cancelled"):
                sample(sequential, latent, cond, end=6, leftover="enable")
        self.assertNotIn("hybrid_latent_shapes", sequential.model_options)
        self.assertNotIn(windows.OFFSET_KEY, sequential.model_options["transformer_options"])
        self.assertTrue(all(torch.count_nonzero(x) == 0 for x in latent["samples"].unbind()))

    def test_control_offsets_reuse_encodes_only_within_one_call(self):
        class VAE:
            def __init__(self):
                self.starts = []
            def spacial_compression_encode(self):
                return 16
            def encode(self, frames):
                self.starts.append(float(frames[0, 0, 0, 0]))
                return torch.full((1, 24, 12, 2, 2), self.starts[-1])
        vae = VAE()
        frames = torch.arange(107).float().view(107, 1, 1, 1).expand(107, 3, 32, 32)
        patcher = control.WindowControl(SimpleNamespace(), vae, frames, None, None, .5, 1., 0.)
        x = [torch.zeros(1, 24, 12, 2, 2), torch.zeros(1, 32, 2, 65)]
        state = control.ControlRun()
        received = []
        def forward(*args, **kwargs):
            received.append(float(patcher.control_latent.mean()))
            return x
        for offset in (0, 34, 68, 0, 34, 68):
            patcher.diffusion_model_wrapper(forward, x, torch.tensor([800.]), None,
                {windows.OFFSET_KEY: offset, control.CACHE_KEY: state})
            self.assertIsNone(patcher.control_latent)
        self.assertEqual(vae.starts, [0, 34, 68])
        self.assertEqual(received, [0, 34, 68, 0, 34, 68])
        self.assertIs(patcher.control_video, frames)
        state = control.ControlRun()
        patcher.diffusion_model_wrapper(forward, x, torch.tensor([800.]), None,
            {windows.OFFSET_KEY: 0, control.CACHE_KEY: state})
        self.assertEqual(vae.starts, [0, 34, 68, 0])

    def test_control_cache_clears_on_cancel_and_is_not_model_owned(self):
        guider = SimpleNamespace(model_options={"transformer_options": {}})
        original = guider.model_options
        states = []
        class Executor:
            class_obj = guider
            def __call__(self):
                state = guider.model_options["transformer_options"][control.CACHE_KEY]
                state.latents[0] = torch.ones(3)
                states.append(state)
                raise RuntimeError("cancelled")
        with self.assertRaisesRegex(RuntimeError, "cancelled"):
            control.control_run(Executor())
        self.assertIs(guider.model_options, original)
        self.assertEqual(states[0].latents, {})


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
