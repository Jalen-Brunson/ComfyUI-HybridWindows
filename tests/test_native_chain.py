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
import comfy.nested_tensor
import comfy.patcher_extension
import comfy.sample
import comfy_extras.nodes_custom_sampler as custom
import nodes as native_nodes
from comfy_extras.nodes_minimax_h3 import EmptyMiniMaxH3LatentAV, MiniMaxH3ImageToVideo, video_latent_t
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
                          "control_offset": transformer_options.get(windows.CONTROL_OFFSET_KEY, 0),
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
        sequential, joint, cond, total, *_ = hybrid.H3HybridWindows.execute(m, 39, 5, groups)
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
            sequential, joint, cond, _, *_ = hybrid.H3HybridWindows.execute(model(), 39, 5, {"positive_0": positive})
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
            sequential, joint, cond, total, *_ = hybrid.H3HybridWindows.execute(
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
            sequential, joint, cond, total, *_ = hybrid.H3HybridWindows.execute(
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
        sequential, joint, cond, total, *_ = hybrid.H3HybridWindows.execute(
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
        sequential, joint, cond, total, *_ = hybrid.H3HybridWindows.execute(m, 39, 5, {"positive_0": prompt()})
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, total)[0]
        with self.assertRaisesRegex(ValueError, "leftover_noise"):
            sample(sequential, latent, cond, end=6)
        with self.assertRaisesRegex(ValueError, "[Dd]isable [Nn]oise|disable add_noise"):
            sample(joint, latent, cond, start=6)
        self.assertEqual(m.model.seen, [])

    def test_prompt_socket_numbers_define_order(self):
        m = model()
        _, _, bound, total, *_ = hybrid.H3HybridWindows.execute(
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
        sequential, _, cond, total, *_ = hybrid.H3HybridWindows.execute(m, 39, 5, {"positive_0": prompt()})
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


def custom_sample(sequential, joint, latent, cond, split=6, steps=8, seed=123, model_for_sigmas=None):
    """The two-SamplerCustomAdvanced chain: one schedule, split at `split`."""
    sigmas = comfy.samplers.calculate_sigmas(
        (model_for_sigmas or sequential).get_model_object("model_sampling"), "simple", steps)
    high, low = custom.SplitSigmas.execute(sigmas, split)
    warm = custom.SamplerCustomAdvanced.execute(
        custom.Noise_RandomNoise(seed), custom.BasicGuider.execute(sequential, cond)[0],
        custom.KSamplerSelect.execute("euler")[0], high, latent)[0]
    if len(low) < 2:
        return warm
    return custom.SamplerCustomAdvanced.execute(
        custom.Noise_EmptyNoise(), custom.BasicGuider.execute(joint, cond)[0],
        custom.KSamplerSelect.execute("euler")[0], low, warm)[0]


def masked_latent(frames, keep_video=0, keep_audio=0):
    """An AV latent whose own mask pins a leading run of rows."""
    latent = EmptyMiniMaxH3LatentAV.execute(32, 32, frames)[0]
    video, audio = latent["samples"].unbind()
    video = video + torch.linspace(0, 1, video.shape[2]).reshape(1, 1, -1, 1, 1)
    audio = audio + .5
    mask = windows.ones_mask(video, audio)
    mask[0][:, :, :keep_video] = 0
    mask[1][:, :, :, :keep_audio] = 0
    return {"samples": comfy.nested_tensor.NestedTensor([video, audio]),
            "noise_mask": comfy.nested_tensor.NestedTensor(mask)}


class NativeSourceTests(unittest.TestCase):
    """The optional inputs: a source master, its pins, resume and per-window noise."""

    def test_custom_sampler_chain_matches_ksampler_advanced_chain(self):
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, 73)[0]
        groups = {f"positive_{i}": prompt(i) for i in range(2)}
        a = model()
        seq_a, joint_a, cond_a, total, *_ = hybrid.H3HybridWindows.execute(a, 39, 5, groups)
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, total)[0]
        first = sample(seq_a, latent, cond_a, end=6, leftover="enable")
        expected = sample(joint_a, first, cond_a, start=6, add_noise="disable")
        b = model()
        seq_b, joint_b, cond_b, _, *_ = hybrid.H3HybridWindows.execute(b, 39, 5, groups)
        actual = custom_sample(seq_b, joint_b, latent, cond_b)
        for got, want in zip(actual["samples"].unbind(), expected["samples"].unbind()):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_source_mask_pins_rows_through_both_stages(self):
        keep_v, keep_a = 4, 6
        source = masked_latent(73, keep_v, keep_a)
        m = model()
        groups = {f"positive_{i}": prompt(i) for i in range(2)}
        seq, joint, cond, total, prepared, report = hybrid.H3HybridWindows.execute(
            m, 39, 5, groups, latent=source)
        self.assertEqual(total, 73)
        result = custom_sample(seq, joint, prepared, cond)
        # Every model call in both stages is told which rows are held.
        self.assertTrue(all("denoise_mask" in call["masks"] for call in m.model.seen))
        video, audio = result["samples"].unbind()
        src_v, src_a = source["samples"].unbind()
        torch.testing.assert_close(video[:, :, :keep_v], src_v[:, :, :keep_v], rtol=2e-6, atol=1e-6)
        torch.testing.assert_close(audio[:, :, :, :keep_a], src_a[:, :, :, :keep_a], rtol=2e-6, atol=1e-6)
        self.assertFalse(torch.allclose(video[:, :, keep_v:], src_v[:, :, keep_v:]))
        self.assertIn("Source pins", report)

    def test_masked_latent_without_the_node_latent_input_is_refused(self):
        source = masked_latent(73, 4, 6)
        seq, joint, cond, _, prepared, _ = hybrid.H3HybridWindows.execute(
            model(), 39, 5, {f"positive_{i}": prompt(i) for i in range(2)})
        self.assertIsNone(prepared)
        with self.assertRaisesRegex(ValueError, "latent"):
            custom_sample(seq, joint, source, cond)

    def test_both_window_offset_keys_are_published(self):
        m = model()
        seq, joint, cond, total, *_ = hybrid.H3HybridWindows.execute(
            m, 39, 5, {f"positive_{i}": prompt(i) for i in range(3)})
        latent = EmptyMiniMaxH3LatentAV.execute(32, 32, total)[0]
        custom_sample(seq, joint, latent, cond)
        native = {call["offset"] for call in m.model.seen}
        mmh3 = {call["control_offset"] for call in m.model.seen}
        self.assertEqual(native, mmh3)
        self.assertEqual(native, {0, 34, 68})

    def test_accepted_prefix_skips_windows_and_keeps_input_rows(self):
        accepted = 39
        source = masked_latent(73)
        m = model()
        groups = {f"positive_{i}": prompt(i) for i in range(2)}
        seq, joint, cond, _, prepared, report = hybrid.H3HybridWindows.execute(
            m, 39, 5, groups, latent=source, accepted_prefix_frames=accepted, start_window=3)
        result = custom_sample(seq, joint, prepared, cond)
        # Window 1 lies inside the prefix: only window 2 is sampled in stage 1.
        warmup_calls = [c for c in m.model.seen if len(c["sigmas"]) == 7]
        self.assertEqual({int(c["prompt"]) for c in warmup_calls}, {1})
        video, audio = result["samples"].unbind()
        src_v, src_a = source["samples"].unbind()
        accepted_v = video_latent_t(accepted)
        torch.testing.assert_close(video[:, :, :accepted_v], src_v[:, :, :accepted_v], rtol=0, atol=0)
        self.assertFalse(torch.allclose(video[:, :, accepted_v:], src_v[:, :, accepted_v:]))
        self.assertIn("Accepted prefix", report)
        # The input latent itself is never written to.
        torch.testing.assert_close(source["samples"].unbind()[0], src_v, rtol=0, atol=0)

    def test_accepted_prefix_off_grid_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"5\+17k"):
            hybrid.H3HybridWindows.execute(model(), 39, 5, {"positive_0": prompt()},
                                           latent=masked_latent(39), accepted_prefix_frames=10)

    def test_per_window_noise_differs_from_the_global_draw(self):
        latents, results = EmptyMiniMaxH3LatentAV.execute(32, 32, 73)[0], []
        groups = {f"positive_{i}": prompt(i) for i in range(2)}
        for mode in ("global", "per_window"):
            m = model()
            seq, joint, cond, _, prepared, _ = hybrid.H3HybridWindows.execute(
                m, 39, 5, groups, latent=latents, noise_mode=mode)
            results.append(custom_sample(seq, joint, prepared, cond)["samples"].unbind()[0])
        self.assertFalse(torch.allclose(results[0], results[1]))

    def test_per_window_noise_matches_a_seeded_draw_per_window(self):
        m = model()
        source = masked_latent(73)
        groups = {f"positive_{i}": prompt(i) for i in range(2)}
        seq, joint, cond, _, prepared, _ = hybrid.H3HybridWindows.execute(
            m, 39, 5, groups, latent=source, noise_mode="per_window", start_window=2)
        run = seq.wrappers[comfy.patcher_extension.WrappersMP.OUTER_SAMPLE]["hybrid_sequential"][0].run
        video, audio = prepared["samples"].unbind()
        spans = run.plan.spans([tuple(video.shape), tuple(audio.shape)])
        custom_sample(seq, joint, prepared, cond, split=8)
        previous = 0
        for index, (v0, v1, a0, a1) in enumerate(spans):
            expected = comfy.sample.prepare_noise(
                comfy.nested_tensor.NestedTensor([video[:, :, v0:v1].clone(),
                                                  audio[:, :, :, a0:a1].clone()]), 123 + 2 + index)
            # At sigma 1 the solver starts at the noise itself; the carried head
            # is the one part the inpaint path replaces before the model sees it.
            carry = max(0, previous - v0)
            drawn = comfy.utils.unpack_latents(
                m.model.seen[index * 8]["x"], [tuple(video[:, :, v0:v1].shape),
                                               tuple(audio[:, :, :, a0:a1].shape)])
            torch.testing.assert_close(drawn[0][:, :, carry:], expected.unbind()[0][:, :, carry:],
                                       rtol=2e-6, atol=1e-6)
            previous = v1

    def test_joint_stage_refuses_a_foreign_or_stale_latent(self):
        m = model()
        source = masked_latent(73, 4, 6)
        groups = {f"positive_{i}": prompt(i) for i in range(2)}
        seq, joint, cond, _, prepared, _ = hybrid.H3HybridWindows.execute(m, 39, 5, groups, latent=source)
        sigmas = comfy.samplers.calculate_sigmas(seq.get_model_object("model_sampling"), "simple", 8)
        high, low = custom.SplitSigmas.execute(sigmas, 6)
        warm = custom.SamplerCustomAdvanced.execute(
            custom.Noise_RandomNoise(123), custom.BasicGuider.execute(seq, cond)[0],
            custom.KSamplerSelect.execute("euler")[0], high, prepared)
        with self.assertRaisesRegex(ValueError, "output"):  # denoised_output, not output
            custom.SamplerCustomAdvanced.execute(
                custom.Noise_EmptyNoise(), custom.BasicGuider.execute(joint, cond)[0],
                custom.KSamplerSelect.execute("euler")[0], low, warm[1])
        fresh_seq, fresh_joint, fresh_cond, _, fresh_prepared, _ = hybrid.H3HybridWindows.execute(
            model(), 39, 5, groups, latent=source)
        with self.assertRaisesRegex(ValueError, "no warmup state"):
            custom.SamplerCustomAdvanced.execute(
                custom.Noise_EmptyNoise(), custom.BasicGuider.execute(fresh_joint, fresh_cond)[0],
                custom.KSamplerSelect.execute("euler")[0], low, warm[0])

    def test_stashed_state_is_the_leftover_latent_rescaled(self):
        """The handoff identity: x_sigma = (1-sigma) * process_latent_in(leftover)."""
        m = model()
        source = masked_latent(73)
        seq, joint, cond, _, prepared, _ = hybrid.H3HybridWindows.execute(
            m, 39, 5, {f"positive_{i}": prompt(i) for i in range(2)}, latent=source)
        sigmas = comfy.samplers.calculate_sigmas(seq.get_model_object("model_sampling"), "simple", 8)
        high, _ = custom.SplitSigmas.execute(sigmas, 6)
        warm = custom.SamplerCustomAdvanced.execute(
            custom.Noise_RandomNoise(123), custom.BasicGuider.execute(seq, cond)[0],
            custom.KSamplerSelect.execute("euler")[0], high, prepared)[0]
        run = seq.wrappers[comfy.patcher_extension.WrappersMP.OUTER_SAMPLE]["hybrid_sequential"][0].run
        leftover = comfy.utils.pack_latents(
            m.model.process_latent_in(warm["samples"]).unbind())[0]
        rebuilt = (1 - float(high[-1])) * leftover
        torch.testing.assert_close(run.stash.raw_state, rebuilt, rtol=2e-6, atol=2e-6)

    def test_all_accepted_needs_no_sampling(self):
        m = model()
        source = masked_latent(39)
        seq, joint, cond, _, prepared, report = hybrid.H3HybridWindows.execute(
            m, 39, 5, {"positive_0": prompt()}, latent=source, accepted_prefix_frames=39)
        result = custom_sample(seq, joint, prepared, cond)
        self.assertEqual(m.model.seen, [])
        for got, want in zip(result["samples"].unbind(), source["samples"].unbind()):
            torch.testing.assert_close(got, want, rtol=0, atol=0)

    def test_cond_set_drives_the_windows(self):
        m = model()
        cond_set = {"conds": [prompt(i) for i in range(3)], "prompts": ["a", "b", "c"]}
        _, _, bound, total, _, _ = hybrid.H3HybridWindows.execute(m, 243, 39, cond_set=cond_set)
        self.assertEqual([float(c[0].mean()) for c in bound], [0, 1, 2])
        self.assertEqual([c[1][windows.WINDOW_KEY] for c in bound], [0, 1, 2])
        self.assertEqual(total, 651)
        with self.assertRaisesRegex(ValueError, "not both"):
            hybrid.H3HybridWindows.execute(m, 243, 39, {"positive_0": prompt()}, cond_set=cond_set)
        with self.assertRaisesRegex(ValueError, "one conditioning per window"):
            hybrid.H3HybridWindows.execute(m, 243, 39)

    def test_short_master_slides_the_last_window_back(self):
        plan = windows.plan_windows(39, 5, 3)
        rigid = video_latent_t(39 + 2 * (39 - 5))
        short = rigid - 2
        shapes = [(1, 24, short, 2, 2), (1, 32, 2, round(windows.frame_at(short) * 40 / 24))]
        spans = plan.spans(shapes)
        self.assertEqual(spans[-1][1], short)
        self.assertEqual(spans[-1][1] - spans[-1][0], plan.length)
        with self.assertRaisesRegex(ValueError, "video latents"):
            plan.spans([(1, 24, rigid + 1, 2, 2), (1, 32, 2, 200)])

    def test_prefix_pin_matches_a_hand_built_mask(self):
        source = masked_latent(73, 2, 3)
        video, audio = source["samples"].unbind()
        pinned = windows.pin_prefix(source["noise_mask"], video, audio, 6, 10)
        vm, am = pinned.unbind()
        self.assertEqual(float(vm[:, :, :6].max()), 0.0)
        self.assertEqual(float(am[:, :, :, :10].max()), 0.0)
        self.assertEqual(float(vm[:, :, 6:].min()), 1.0)
        # keep-wins: the master's own pins are still there.
        original = source["noise_mask"].unbind()[0]
        self.assertEqual(float(original[:, :, :2].max()), 0.0)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]])
