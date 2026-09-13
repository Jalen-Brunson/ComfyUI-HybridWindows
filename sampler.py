"""Experimental causal warmup followed by joint-window sampling."""

import importlib
import logging

from .dependencies import _mmh3_module
from .windows import tail_context_rows

import torch
import comfy.utils
from comfy.context_windows import create_prepare_sampling_wrapper
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from comfy_extras.nodes_custom_sampler import Guider_Basic, SamplerCustomAdvanced

# Shared with the native two-node chain; imported here so existing callers keep
# reaching them as `sampler.EulerSegment` / `sampler.FixedNoise`.
from .segment import EulerSegment, FixedNoise, SolverSegment  # noqa: F401

MMH3CondSet = io.Custom("MMH3_COND_SET")
MASK_MODES = ["max", "min", "mean", "last"]


def _latent_parts(latent, video, audio, mask=None):
    out = {**latent, "samples": NestedTensor([video, audio])}
    out.pop("noise_mask", None)
    if mask is not None:
        out["noise_mask"] = mask
    return out


def _write_new(destination, source, v0, v1, a0, a1, previous_v, previous_a):
    """The first window to sample a frame owns its state and original noise."""
    dv, da = destination
    sv, sa = source
    first_v, first_a = max(v0, previous_v), max(a0, previous_a)
    dv[:, :, first_v:v1] = sv[:, :, first_v - v0:].to(dv)
    da[:, :, :, first_a:a1] = sa[:, :, :, first_a - a0:].to(da)


class MMH3HybridWindowSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MMH3HybridWindowSampler",
            display_name="H3 Hybrid Window Sampler (experimental)",
            category="sampling/hybrid", is_experimental=True,
            description=("Run the first steps sequentially with video carry pinned, then finish "
                         "the same noisy state with joint windows. Any solver, CFG 1. "
                         "An accepted prefix stays fixed in both phases. "
                         "A full sequential_steps count is the sequential baseline. Quality is unvalidated."),
            inputs=[
                io.Model.Input("model"), io.Noise.Input("noise"),
                io.Sampler.Input("sampler"), io.Sigmas.Input("sigmas"),
                MMH3CondSet.Input("cond_set"), io.Latent.Input("latent"),
                io.Int.Input("window_frames", default=243, min=5, max=100000, step=17),
                io.Int.Input("overlap_frames", default=39, min=0, max=100000, step=17),
                io.Int.Input("sequential_steps", default=6, min=1, max=10000,
                             tooltip="Steps before switching to joint windows. Equal to total steps gives a sequential baseline."),
                io.Combo.Input("accumulator_device", options=["gpu", "cpu"], default="gpu"),
                io.Combo.Input("denoise_mask_mode", options=MASK_MODES, default="max"),
                io.Mask.Input("denoise_mask", optional=True),
                io.Mask.Input("audio_denoise_mask", optional=True),
                io.Int.Input("accepted_prefix_frames", default=0, min=0, max=1000000,
                             optional=True, tooltip="Finished frames at the start of this input to preserve, including audio. With H3ContinueMaster, wire one chunk's frame count on resume and 0 on fresh runs."),
                io.Int.Input("start_window", default=0, min=0, max=100000,
                             optional=True, tooltip="Global index of this input's first window, for per-window seeds. Wire the resume index shift; control offsets stay relative to this run's loaded control."),
                io.Int.Input("feather_latents", default=0, min=0, max=32, optional=True,
                             tooltip="Sequential-phase carry seam only, VIDEO only. Linear mask ramp over N latents AFTER each window's carried region, easing back to full generation instead of stepping. 0 disables. The accepted prefix is never feathered. Re-test knob restored 2026-09-12; keep at 0 for renders."),
                io.Combo.Input("overlap_pin", options=["full", "tail_custom"], default="full", optional=True,
                               tooltip="full preserves the entire warmup overlap. tail_custom pins only context_frames at its end. The full overlap still reaches the model; accepted output stays fixed."),
                io.Int.Input("context_frames", default=5, min=0, max=3600, optional=True,
                             tooltip="Used with overlap_pin=tail_custom. Pinned video frames at the end of the warmup overlap; 0 releases video carry. Rounds up to whole latent rows and caps at the overlap. Audio carry and joint windows stay unchanged."),
            ],
            outputs=[io.Latent.Output(display_name="latent"), io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, noise, sampler, sigmas, cond_set, latent,
                window_frames, overlap_frames, sequential_steps=6,
                accumulator_device="gpu", denoise_mask_mode="max",
                denoise_mask=None, audio_denoise_mask=None,
                accepted_prefix_frames=0, start_window=0, feather_latents=0,
                overlap_pin="full", context_frames=5):
        common = _mmh3_module("common")
        frame_at_latent = common.frame_at_latent
        frames_to_latents = common.frames_to_latents
        latents_to_frames = common.latents_to_frames
        unpack_av = common.unpack_av
        nodes_joint_sampler = importlib.import_module(".joint", __package__)
        MMH3JointContextHandler = nodes_joint_sampler.MMH3JointContextHandler
        bind_window_conditioning = nodes_joint_sampler.bind_window_conditioning
        prepare_masks = nodes_joint_sampler.prepare_masks
        nodes_looping_sampler = _mmh3_module("nodes_looping_sampler")
        _carry_mask = nodes_looping_sampler._carry_mask
        _chunk_guider = nodes_looping_sampler._chunk_guider
        _chunk_noise = nodes_looping_sampler._chunk_noise
        _sliced_mask = nodes_looping_sampler._sliced_mask
        _split_mask = nodes_looping_sampler._split_mask
        _strip_guide_keys = nodes_looping_sampler._strip_guide_keys
        nodes_windows = _mmh3_module("nodes_windows")
        _audio_index_at = nodes_windows._audio_index_at
        _plan = nodes_windows._plan
        _window_frame_spans = nodes_windows._window_frame_spans

        steps = len(sigmas) - 1
        if not 1 <= sequential_steps <= steps:
            raise ValueError(f"sequential_steps must be between 1 and {steps} for this schedule.")
        if not bool(torch.isfinite(sigmas).all()) or not bool((sigmas[:-1] > sigmas[1:]).all()) or float(sigmas[-1]) != 0:
            raise ValueError("Hybrid sampling needs finite descending sigmas ending at zero.")
        SolverSegment(sampler)  # Validate solver before encoding any sampling state.
        if "context_handler" in model.model_options:
            raise ValueError("Connect the model before any Context Windows node; the hybrid sampler owns windowing.")
        video, audio = unpack_av(latent)
        length, overlap, total_f, total_t, windows = _plan(
            latents_to_frames(video.shape[2]), window_frames, overlap_frames, "standard_static")
        if total_t != video.shape[2] or any(w.index_list[0] % 5 for w in windows):
            raise ValueError("Hybrid sampling needs H3-aligned latent lengths and window starts.")
        bound = bind_window_conditioning(cond_set, windows)
        prepared = prepare_masks(latent, denoise_mask, audio_denoise_mask, denoise_mask_mode)
        in_v, in_a = _split_mask(prepared)
        accepted_f = int(accepted_prefix_frames)
        accepted_v = frames_to_latents(accepted_f) if accepted_f > 0 else 0
        if accepted_f < 0 or accepted_f > total_f or frame_at_latent(accepted_v) != accepted_f:
            raise ValueError("accepted_prefix_frames must be 0 or 5+17k frames within this input.")
        if int(start_window) < 0:
            raise ValueError("start_window must be a nonnegative global window index.")
        if overlap_pin == "tail_custom" and in_v is not None and torch.any(in_v[:, :, accepted_v:] != 1):
            raise ValueError("Partial video pinning needs full-frame regeneration outside the accepted prefix.")
        accepted_a = _audio_index_at(accepted_v, total_t, audio.shape[3])
        if accepted_v:
            prepared["noise_mask"] = _carry_mask(
                video, audio, accepted_v, accepted_a, 1.0, 1.0,
                in_v, in_a, 0, total_t, 0, audio.shape[3])
            in_v, in_a = _split_mask(prepared)
            if accepted_v == total_t:
                return io.NodeOutput(_latent_parts(prepared, video.clone(), audio.clone()),
                                     f"All {total_f} frames accepted; no sampling needed.")
        clean_v, clean_a = video.clone(), audio.clone()
        total_a = audio.shape[3]
        joint_steps = steps - sequential_steps
        if joint_steps:
            states = (torch.empty_like(video, device="cpu", dtype=torch.float32),
                      torch.empty_like(audio, device="cpu", dtype=torch.float32))
            noises = tuple(torch.empty_like(x) for x in states)
        guider = Guider_Basic(model)
        previous_v = previous_a = 0
        spans = _window_frame_spans(windows, total_f)
        solver_name = getattr(sampler.sampler_function, "__name__", "unknown").replace("sample_", "")
        lines = [f"Hybrid: {len(windows)} windows / {total_f} frames; "
                 f"{sequential_steps} sequential + {joint_steps} joint {solver_name} steps.",
                 f"Switch sigma {float(sigmas[sequential_steps]):.9f}; frame spans {spans}.",
                 f"Seed {noise.seed} + global window index (starts at {int(start_window)}). "
                 "Video carry strength 1; audio carry strength 1."
                 + (f" Carry feather {int(feather_latents)} latents." if feather_latents else "")]
        if accepted_v:
            lines.append(f"Accepted prefix: {accepted_f} frames / {accepted_v} video and {accepted_a} audio latents. "
                         "Fixed in both phases; accepted windows skip sequential sampling and remain joint context.")

        context_counts = set()
        for i, window in enumerate(windows):
            v0, v1 = window.index_list[0], window.index_list[-1] + 1
            a0, a1 = _audio_index_at(v0, total_t, total_a), _audio_index_at(v1, total_t, total_a)
            sub_v, sub_a = clean_v[:, :, v0:v1].clone(), clean_a[:, :, :, a0:a1].clone()
            carried = max(0, previous_v - v0)
            if carried:
                mask = _carry_mask(sub_v, sub_a, carried, max(0, previous_a - a0),
                                   1.0, 1.0, in_v, in_a, v0, v1, a0, a1)
                if feather_latents:
                    video_mask = mask.unbind()[0]
                    count = min(int(feather_latents), video_mask.shape[2] - carried)
                    if count > 0:
                        ramp = torch.linspace(0, 1, count + 1, device=video_mask.device)[1:]
                        video_mask[:, :, carried:carried + count] = torch.minimum(
                            video_mask[:, :, carried:carried + count], ramp.reshape(1, 1, -1, 1, 1))
                if overlap_pin == "tail_custom":
                    held = tail_context_rows(v0, previous_v, int(context_frames))
                    mask.unbind()[0][:, :, :carried - held] = 1
                    if v1 > accepted_v:
                        context_counts.add(frame_at_latent(previous_v) - frame_at_latent(previous_v - held))
            elif in_v is not None or in_a is not None:
                mask = _sliced_mask(sub_v, sub_a, in_v, in_a, v0, v1, a0, a1)
            else:
                mask = None
            chunk = _latent_parts(prepared, sub_v, sub_a, mask)
            window_noise = _chunk_noise(noise, int(start_window) + i)
            fixed = FixedNoise(window_noise.generate_noise(chunk), window_noise.seed)
            if v1 <= accepted_v:
                if joint_steps:
                    # Fully pinned rows never reach the model as this raw state:
                    # native inpainting supplies the accepted source at every step.
                    pinned_state = [torch.zeros_like(x) for x in (sub_v, sub_a)]
                    _write_new(states, pinned_state, v0, v1, a0, a1, previous_v, previous_a)
                    _write_new(noises, fixed.samples.unbind(), v0, v1, a0, a1, previous_v, previous_a)
                previous_v, previous_a = v1, a1
                logging.info("[MMH3HybridWindowSampler] accepted window %d: sequential pass skipped", int(start_window) + i + 1)
                continue
            positive = _strip_guide_keys(cond_set["conds"][i], f"hybrid window {i}")
            chunk_guider = _chunk_guider(guider, positive, frame_at_latent(v0))
            logging.info("[MMH3HybridWindowSampler] sequential window %d/%d, steps 0..%d",
                         i + 1, len(windows), sequential_steps)
            segment = SolverSegment(sampler, capture=bool(joint_steps))
            result = SamplerCustomAdvanced.execute(
                fixed, chunk_guider, segment, sigmas[:sequential_steps + 1], chunk)
            if joint_steps:
                shapes = [sub_v.shape, sub_a.shape]
                state_parts = comfy.utils.unpack_latents(segment.final_state, shapes)
                _write_new(states, state_parts, v0, v1, a0, a1, previous_v, previous_a)
                _write_new(noises, fixed.samples.unbind(), v0, v1, a0, a1, previous_v, previous_a)
                prediction = NestedTensor(comfy.utils.unpack_latents(segment.final_prediction, shapes))
                prediction = model.model.process_latent_out(prediction)
                dv, da = prediction.unbind()
            else:
                dv, da = unpack_av(result[1])
            # Finished mode matches native sequential write-back precision. In hybrid
            # mode these are provisional conditioning only, never the resumed state.
            write_v, write_a = max(v0, accepted_v), max(a0, accepted_a)
            if overlap_pin == "tail_custom":
                # Released overlap is temporary context, never a replacement for its first owner's prediction.
                write_v = max(write_v, previous_v)
            clean_v[:, :, write_v:v1] = dv[:, :, write_v-v0:].to(clean_v)
            clean_a[:, :, :, write_a:a1] = da[:, :, :, write_a-a0:].to(clean_a)
            previous_v, previous_a = v1, a1

        if not joint_steps:
            output = _latent_parts(prepared, clean_v, clean_a)
        else:
            patched = model.clone()
            patched.model_options["context_handler"] = MMH3JointContextHandler(
                windows, length, overlap, accumulator_device)
            create_prepare_sampling_wrapper(patched)
            joint_guider = Guider_Basic(patched)
            joint_guider.set_conds(bound)
            raw_state, _ = comfy.utils.pack_latents(states)
            resume_sampler = SolverSegment(sampler, initial_state=raw_state)
            fixed = FixedNoise(NestedTensor(noises), noise.seed)
            logging.info("[MMH3HybridWindowSampler] joint finish: %d windows, %d steps, sigma %.9f",
                         len(windows), joint_steps, float(sigmas[sequential_steps]))
            # The original input supplies source pins and mask labels. The solver
            # starts from raw_state; provisional clean carry is deliberately absent.
            output = SamplerCustomAdvanced.execute(
                fixed, joint_guider, resume_sampler, sigmas[sequential_steps:], prepared)[1]
            lines.append("Resumed raw float32 x_sigma without fresh noise. Generated carry released; source and accepted-prefix masks retained.")
        if accepted_v:
            ov, oa = unpack_av(output)
            # Assemble the accepted values directly, including any round-off in
            # model-space scaling. Only the continuation comes from the sampler.
            output = _latent_parts(prepared,
                torch.cat([video[:, :, :accepted_v].to(ov), ov[:, :, accepted_v:]], dim=2),
                torch.cat([audio[:, :, :, :accepted_a].to(oa), oa[:, :, :, accepted_a:]], dim=3))
        lines.append(f"Video overlap pin: {overlap_pin}.")
        if overlap_pin == "tail_custom":
            effective = ", ".join(str(count) for count in sorted(context_counts)) or "0 (no sampled carry)"
            lines.append(f"Pinned video context: requested {int(context_frames)} frames; effective {effective} frames. "
                         "Full overlap remains visible; audio carry and joint windows are unchanged.")
        lines.append("Experimental: continuity and texture improvement are not established.")
        report = "\n".join(lines)
        logging.info("[MMH3HybridWindowSampler] %s", report)
        return io.NodeOutput(output, report)
