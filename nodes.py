"""Hybrid windows through ordinary native sampler MODEL/LATENT connections.

Two stock sampler nodes do the sampling: the first runs the windows in order
with the overlap carried, the second finishes them together. Everything this
module adds rides on model wrappers, so the graph stays native.

The optional inputs (a cond set, a source master, masks, an accepted prefix,
per-window noise) exist so a production graph -- v2v inpainting with an
accepted-prefix resume -- can run on this chain instead of the single-node
sampler. With none of them connected the node behaves exactly as before.
"""

import importlib
import logging
import math

import torch

import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.sample
import comfy.utils
from comfy.nested_tensor import NestedTensor
from comfy.patcher_extension import WrappersMP
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import video_latent_t

from .segment import EulerSegment
from .windows import (JointWindows, WINDOW_KEY, audio_index_at, frame_at, pin_prefix,
                      plan_windows, select_conditioning, slice_av, window_options, write_new)

# Matched by name, so MMH3Tools is never imported for the schema.
MMH3CondSet = io.Custom("MMH3_COND_SET")
MASK_MODES = ["max", "min", "mean", "last"]
NOISE_MODES = ["global", "per_window"]


def add_outermost_wrapper(patcher, kind, key, wrapper):
    """Register `wrapper` so it runs outside every other wrapper of `kind`.

    Core calls wrappers in registration order, first registered outermost. The
    window loop has to be the outermost OUTER_SAMPLE wrapper: previews, caches
    and other per-sample hooks registered anywhere in the graph then enter once
    per window, with the window's own noise, sigmas and latent shapes, instead
    of once around the whole loop where they never see the inner samples.
    """
    slot = patcher.wrappers.setdefault(kind, {})
    others = {k: v for k, v in slot.items() if k != key}
    slot.clear()
    slot[key] = [wrapper]
    slot.update(others)


def latent_signature(packed, shapes):
    """Cheap identity of a packed latent, for the stage-1 -> stage-2 handoff.

    Specific enough to catch the mistakes that matter -- a stale run, another
    chain's latent, `denoised_output` instead of `output` -- without copying a
    whole master to hash it.
    """
    flat = packed.detach().flatten()
    probe = flat[::max(1, flat.numel() // 4096)].to(torch.float64)
    return {"shapes": tuple(tuple(s) for s in shapes), "shape": tuple(packed.shape),
            "sum": float(flat.to(torch.float64).sum()), "probe": float(probe.abs().sum())}


def signatures_match(left, right, tol=1e-6):
    if left["shapes"] != right["shapes"] or left["shape"] != right["shape"]:
        return False
    return all(math.isclose(left[key], right[key], rel_tol=tol, abs_tol=1e-6)
               for key in ("sum", "probe"))


class HybridStash:
    """What the joint stage needs from the warm-up that a LATENT cannot carry."""

    def __init__(self, signature, sigma_end, seed, raw_state, noise):
        self.signature = signature
        self.sigma_end = sigma_end
        self.seed = seed
        self.raw_state = raw_state
        self.noise = noise


class HybridRun:
    """State shared by the two stages of one H3HybridWindows execution.

    `ModelPatcher.clone` copies the wrapper lists but not the wrapper objects,
    so the instance the sequential model carries and the one the joint model
    carries are this same object, through every downstream clone.
    """

    def __init__(self, plan, total_steps=8, noise_mode="global", start_window=0):
        self.plan = plan
        self.total_steps = total_steps
        self.noise_mode = noise_mode
        self.start_window = int(start_window)
        self.active = False
        self.source = None
        self.accepted_v = 0
        self.accepted_a = 0
        self.all_accepted = False
        self.expects_mask = False
        self.stash = None

    def splice_accepted(self, packed, shapes):
        """Accepted rows come from the input, not from the sampler."""
        if not self.accepted_v or self.source is None:
            return packed
        parts = comfy.utils.unpack_latents(packed, shapes)
        video, audio = [p.clone() for p in parts]
        video[:, :, :self.accepted_v] = self.source[0][:, :, :self.accepted_v].to(video)
        audio[:, :, :, :self.accepted_a] = self.source[1][:, :, :, :self.accepted_a].to(audio)
        return comfy.utils.pack_latents([video, audio])[0]


class SequentialWindows:
    """Stage 1: each window in order, the previous window's overlap pinned."""

    def __init__(self, run, total_steps=None):
        # `total_steps` is accepted for callers that still build a plan-only run.
        self.run = run if isinstance(run, HybridRun) else HybridRun(run, total_steps or 8)

    def _window_noise(self, source, span, noise_parts, seed, index, like):
        """(packed noise, seed) for one window.

        `per_window` reproduces the single-node sampler's draws: one generator
        seeded base + global window index, video then audio, on the window's own
        shapes. `global` slices the one draw the sampler node made, which is what
        this chain has always done.
        """
        run = self.run
        if run.noise_mode == "per_window":
            window_seed = int(seed or 0) + run.start_window + index
            drawn = comfy.sample.prepare_noise(NestedTensor(list(source)), window_seed)
            return comfy.utils.pack_latents(drawn.unbind())[0].to(like), window_seed
        return comfy.utils.pack_latents(slice_av(noise_parts, span))[0], seed

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                 callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        run = self.run
        if sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler:
            raise ValueError("Hybrid warmup currently supports plain Euler; select euler in the sampler node.")
        if sampler.extra_options.get("s_churn", 0) or sampler.inpaint_options.get("random", False):
            raise ValueError("Hybrid warmup needs Euler without churn or random inpaint noise.")
        if denoise_mask is not None and torch.any(denoise_mask != 1) and run.source is None:
            raise ValueError(
                "This latent pins rows with a noise mask. Connect the master to H3 Hybrid "
                "Windows' `latent` input and sample its `latent` output, so the node can hold "
                "the pinned rows through both stages.")
        finished = float(sigmas[-1]) == 0 and len(sigmas) - 1 == run.total_steps
        if not finished and not 0 < float(sigmas[-1]) < float(sigmas[0]):
            raise ValueError("Warmup must stop before the final step: split the sigmas (or enable "
                             "return_with_leftover_noise) so the joint stage finishes them.")
        if torch.count_nonzero(noise) == 0:
            raise ValueError("Connect Random Noise to the warmup sampler (or enable add_noise).")
        guider = executor.class_obj
        original_conds, original_options = guider.conds, guider.model_options
        shapes = latent_shapes
        spans = self.run.plan.spans(shapes)
        if run.active and run.source is not None:
            expected = latent_signature(comfy.utils.pack_latents(
                [x.to(latent_image) for x in run.source])[0], shapes)
            if not signatures_match(expected, latent_signature(latent_image, shapes)):
                raise ValueError(
                    "The warmup's latent is not this node's `latent` output. Connect H3 Hybrid "
                    "Windows' `latent` output to the warmup sampler so both stages share one source.")
        if run.all_accepted:
            # Nothing to add: hand the master straight back and let the joint
            # stage pass it through, rather than sampling rows that are pinned.
            run.stash = HybridStash(latent_signature(latent_image, shapes),
                                    float(sigmas[-1]), seed, None, None)
            logging.info("[Hybrid Windows] every window is accepted; no sampling needed.")
            return latent_image
        clean = [x.clone() for x in comfy.utils.unpack_latents(latent_image, shapes)]
        noise_parts = comfy.utils.unpack_latents(noise, shapes)
        mask_parts = (comfy.utils.unpack_latents(denoise_mask, shapes)
                      if denoise_mask is not None else None)
        output = [torch.empty_like(x, dtype=torch.float32) for x in clean]
        stash_state = run.active and not finished
        if stash_state:
            states = [torch.empty_like(x, device="cpu", dtype=torch.float32) for x in clean]
            noises = [torch.empty_like(x, device="cpu", dtype=torch.float32) for x in clean]
        previous = (0, 0)
        steps = len(sigmas) - 1
        total_calls = len(spans) * steps
        try:
            for index, span in enumerate(spans):
                comfy.model_management.throw_exception_if_processing_interrupted()
                source = [x.clone() for x in slice_av(clean, span)]
                # The master's own pins for this span, then the carry on top of
                # them: a row either side wants held stays held.
                mask = ([m.clone() for m in slice_av(mask_parts, span)] if mask_parts is not None
                        else [torch.ones_like(x) for x in source])
                carry_v, carry_a = max(0, previous[0]-span[0]), max(0, previous[1]-span[2])
                mask[0][:, :, :carry_v] = 0
                mask[1][:, :, :, :carry_a] = 0
                source_packed, sub_shapes = comfy.utils.pack_latents(source)
                noise_packed, window_seed = self._window_noise(
                    source, span, noise_parts, seed, index, latent_image)
                # CFGGuider normally prepares masks before OUTER_SAMPLE. These
                # carry masks are created here, so honour that same device contract.
                mask_packed = (comfy.utils.pack_latents(mask)[0].to(guider.model_patcher.load_device)
                               if carry_v or carry_a or mask_parts is not None else None)
                if span[1] <= run.accepted_v:
                    # Every row is pinned: the model would only be shown the
                    # source and hand it straight back. Its state is never read
                    # either -- the joint stage re-injects these rows each step.
                    blank = [torch.zeros_like(x) for x in source]
                    write_new(output, blank, span, previous)
                    if stash_state:
                        write_new(states, blank, span, previous)
                        write_new(noises, comfy.utils.unpack_latents(noise_packed, sub_shapes),
                                  span, previous)
                    previous = (span[1], span[3])
                    logging.info("[Hybrid Windows] accepted window %d/%d: warmup skipped",
                                 index + 1, len(spans))
                    continue
                guider.conds = {key: [dict(c) for c in select_conditioning(group, index)]
                                for key, group in original_conds.items()}
                guider.model_options = window_options(original_options, span)
                last_prediction = None
                # Accepted rows keep the source; only fresh rows take a prediction.
                write_v, write_a = max(span[0], run.accepted_v), max(span[2], run.accepted_a)

                def splice(target, prediction):
                    target[0][:, :, write_v:span[1]] = prediction[0][:, :, write_v-span[0]:].to(target[0])
                    target[1][:, :, :, write_a:span[3]] = prediction[1][:, :, :, write_a-span[2]:].to(target[1])

                def capture(step, x0, x, total_steps):
                    nonlocal last_prediction
                    # Native process_latent_out also removes H3's carried-audio scale.
                    last_prediction = guider.inner_model.process_latent_out(x0).to(
                        device=latent_image.device, dtype=torch.float32).clone()
                    if callback is not None:
                        # The sampler node's callback (progress bar, previews, the
                        # denoised output) belongs to the whole master: core unpacks
                        # it with the master's shapes. Hand it the master with this
                        # window's prediction spliced in, every step.
                        preview = [c.clone() for c in clean]
                        splice(preview, comfy.utils.unpack_latents(last_prediction, sub_shapes))
                        packed = comfy.utils.pack_latents(preview)[0]
                        callback(index*steps + step, packed, packed, total_calls)

                logging.info("[Hybrid Windows] sequential window %d/%d, %d steps", index + 1, len(spans), steps)
                segment = EulerSegment(sampler, capture=stash_state)
                result = executor(noise_packed, source_packed, segment, sigmas,
                                  mask_packed, capture, disable_pbar, window_seed, latent_shapes=sub_shapes)
                # Keep the native leftover-noise representation, owned by the first
                # window covering each position. No x0 substitution or fresh noise.
                write_new(output, comfy.utils.unpack_latents(result, sub_shapes), span, previous)
                if stash_state:
                    write_new(states, comfy.utils.unpack_latents(segment.final_state, sub_shapes),
                              span, previous)
                    write_new(noises, comfy.utils.unpack_latents(noise_packed, sub_shapes),
                              span, previous)
                splice(clean, comfy.utils.unpack_latents(last_prediction, sub_shapes))
                previous = (span[1], span[3])
        finally:
            guider.conds, guider.model_options = original_conds, original_options
        # At the full split the second stock sampler is an exact passthrough.
        # Return completed carry, as in sequential generation; partial splits
        # still pass the first owner's noisy state to joint sampling.
        if finished:
            return run.splice_accepted(comfy.utils.pack_latents(clean)[0], shapes)
        leftover = comfy.utils.pack_latents(output)[0]
        if stash_state:
            run.stash = HybridStash(
                signature=latent_signature(leftover, shapes),
                sigma_end=float(sigmas[-1]), seed=seed,
                raw_state=comfy.utils.pack_latents(states)[0],
                noise=comfy.utils.pack_latents(noises)[0])
        return leftover


class JointStage:
    """Stage 2: one global Euler step per sigma, windows fused inside it."""

    def __init__(self, run=None):
        self.run = run

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                 callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        run = self.run
        active = run is not None and run.active
        if len(sigmas) < 2:
            return latent_image  # full sequential split: the warmup already finished
        if torch.count_nonzero(noise):
            raise ValueError("Connect Disable Noise to the joint sampler (or disable add_noise) "
                             "and feed it the warmup's `output` latent.")
        if denoise_mask is not None and torch.any(denoise_mask != 1) and not active:
            raise ValueError("The joint stage requires an unmasked warmup output.")
        if sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler:
            raise ValueError("Use plain euler in both sampler nodes.")
        if float(sigmas[0]) >= 1:
            raise ValueError("Start the joint stage where the warmup stopped: split the sigmas at "
                             "the warmup's step count.")
        guider = executor.class_obj
        original = guider.model_options
        guider.model_options = {**original, "hybrid_latent_shapes": latent_shapes}
        try:
            if not active:
                return executor(noise, latent_image, sampler, sigmas, denoise_mask, callback,
                                disable_pbar, seed, latent_shapes=latent_shapes)
            stash = run.stash
            if stash is None:
                raise ValueError(
                    "The joint stage has no warmup state. Run both samplers from one queue of "
                    "this H3 Hybrid Windows node; a restart between them loses the captured state.")
            if not signatures_match(stash.signature, latent_signature(latent_image, latent_shapes)):
                raise ValueError(
                    "The joint stage's latent is not this warmup's `output` (stale, from another "
                    "chain, or `denoised_output`). Connect output slot 0 of the warmup sampler.")
            if abs(float(sigmas[0]) - stash.sigma_end) > 1e-7:
                raise ValueError(
                    f"The joint stage starts at {float(sigmas[0]):.9f} but the warmup stopped at "
                    f"{stash.sigma_end:.9f}. Split one sigma schedule at the warmup's step count.")
            if run.all_accepted:
                return latent_image
            if run.expects_mask and denoise_mask is None:
                raise ValueError(
                    "The joint stage lost the warmup's noise mask. Connect the warmup sampler's "
                    "`output` latent directly, with no latent node in between.")
            # The solver resumes the raw state the warmup left; the model keeps
            # seeing the CLEAN source, so pinned rows are re-injected from it
            # rather than from a half-denoised latent.
            raw = stash.raw_state.to(device=latent_image.device, dtype=latent_image.dtype)
            clean = latent_image
            if run.source is not None:
                clean = comfy.utils.pack_latents([x.to(latent_image) for x in run.source])[0]
            out = executor(stash.noise.to(device=latent_image.device, dtype=latent_image.dtype),
                           clean, EulerSegment(sampler, initial_state=raw), sigmas, denoise_mask,
                           callback, disable_pbar, stash.seed, latent_shapes=latent_shapes)
            return run.splice_accepted(out, latent_shapes)
        finally:
            guider.model_options = original


# Kept as a name for graphs and tests built against the previous release.
validate_joint = JointStage()


def prepare_joint(executor, model, noise_shape, conds, **kwargs):
    options = kwargs["model_options"]
    shapes = options["hybrid_latent_shapes"]
    plan = options["context_handler"].plan
    spans = plan.spans(shapes)
    # Budget the largest AV window instead of a fictitious full-timeline forward.
    video_size = shapes[0][1] * plan.length * math.prod(shapes[0][3:])
    audio_size = math.prod(shapes[1][1:3]) * max(a1-a0 for _, _, a0, a1 in spans)
    return executor(model, [noise_shape[0], 1, video_size+audio_size], conds, **kwargs)


def bind_conditioning(cond_set, prompts):
    """One conditioning per window, tagged so both stages select it."""
    groups = [prompts[name] for name in sorted(prompts or {},
                                               key=lambda name: int(name.rsplit("_", 1)[1]))]
    if cond_set is not None:
        if groups:
            raise ValueError("Connect either `cond_set` or the per-window prompt sockets, not both.")
        groups = list(cond_set.get("conds") or [])
        if not groups:
            raise ValueError("`cond_set` carries no conditioning; give the reference encoder one "
                             "prompt per window.")
    if not groups:
        raise ValueError("Connect one conditioning per window: a `cond_set`, or the prompt sockets.")
    bound = []
    for index, group in enumerate(groups):
        if not group:
            raise ValueError(f"Connect conditioning for prompt {index + 1}.")
        for tokens, metadata in group:
            # Native image guides use positions within this conditioning's
            # window. Window selection keeps them scoped in both stages.
            bound.append([tokens, {**metadata, WINDOW_KEY: index}])
    return bound, len(groups)


def prepare_source(run, latent, denoise_mask, audio_denoise_mask, mask_mode, accepted_frames):
    """Validate the master, compose its noise mask, and arm the run.

    Returns the latent the samplers should be given, or None when the node is
    driven the original way (an empty latent wired straight to the sampler).
    """
    accepted_frames = int(accepted_frames or 0)
    has_mask = denoise_mask is not None or audio_denoise_mask is not None
    if latent is None:
        if has_mask or accepted_frames:
            raise ValueError(
                "Masks and an accepted prefix need the master: connect it to `latent` and take "
                "the samplers' latent from this node's `latent` output.")
        # Per-window noise still needs the two stages to agree on the draws.
        run.active = run.noise_mode == "per_window" or run.start_window > 0
        return None
    samples = latent.get("samples")
    parts = samples.unbind() if getattr(samples, "is_nested", False) else []
    if len(parts) != 2:
        raise ValueError("H3 hybrid windows need a video+audio latent (Empty MiniMax H3 AV Latent, "
                         "MMH3 Pack AV, or a continuation master).")
    video, audio = parts
    total_v, total_a = video.shape[2], audio.shape[3]
    run.plan.spans([tuple(video.shape), tuple(audio.shape)])  # geometry, with a helpful error
    prepared = dict(latent)
    if has_mask:
        # Pixel masks land on the latent grid through MMH3's reduction, so this
        # chain pins exactly the rows the single-node sampler would.
        prepare_masks = importlib.import_module(".joint", __package__).prepare_masks
        prepared = prepare_masks(latent, denoise_mask, audio_denoise_mask, mask_mode)
    accepted_v = 0
    if accepted_frames:
        accepted_v = video_latent_t(accepted_frames)
        if frame_at(accepted_v) != accepted_frames or accepted_frames > frame_at(total_v):
            raise ValueError(
                f"accepted_prefix_frames must be 0 or 5+17k frames inside this latent "
                f"(up to {frame_at(total_v)}); got {accepted_frames}.")
    accepted_a = audio_index_at(accepted_v, total_v, total_a)
    if accepted_v:
        prepared = dict(prepared)
        prepared["noise_mask"] = pin_prefix(prepared.get("noise_mask"), video, audio,
                                            accepted_v, accepted_a)
    run.active = True
    run.source = [video.clone(), audio.clone()]
    run.accepted_v, run.accepted_a = accepted_v, accepted_a
    run.all_accepted = accepted_v >= total_v
    run.expects_mask = prepared.get("noise_mask") is not None
    return prepared


def run_report(run, count, prepared):
    lines = [f"Hybrid (native): {count} window(s) of {run.plan.window_frames} frames, "
             f"{run.plan.overlap_frames} overlap; {run.total_steps} total steps."]
    if prepared is None:
        lines.append("Empty-latent mode: no source pins, no accepted prefix.")
    else:
        video, audio = prepared["samples"].unbind()
        spans = run.plan.spans([tuple(video.shape), tuple(audio.shape)])
        lines.append(f"Master {frame_at(video.shape[2])} frames / {video.shape[2]} video and "
                     f"{audio.shape[3]} audio latents; window spans {spans}.")
        mask = prepared.get("noise_mask")
        lines.append("Source pins: " + ("none." if mask is None else
                     "%.1f%% of video rows held." % (100.0 * float((mask.unbind()[0] < 1).float().mean()))))
    if run.accepted_v:
        lines.append(f"Accepted prefix {frame_at(run.accepted_v)} frames "
                     f"({run.accepted_v} video / {run.accepted_a} audio latents): fixed in both "
                     "stages, its windows skip the warmup, and the output takes them from the input.")
    lines.append("Noise: " + ("per window, seed + %d + window index." % run.start_window
                              if run.noise_mode == "per_window" else "one draw sliced per window."))
    lines.append("Split the sigmas at the warmup's step count; the joint stage takes the warmup's "
                 "`output` latent with Disable Noise.")
    lines.append("Experimental: continuity and texture improvement are not established.")
    return "\n".join(lines)


class H3HybridWindows(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3HybridWindows", display_name="H3 Hybrid Windows",
            category="sampling/hybrid", is_experimental=True,
            description="Sequential overlap carry followed by joint prediction fusion through two native sampler nodes.",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("window_frames", default=243, min=22, max=3600, step=17),
                io.Int.Input("overlap_frames", default=39, min=5, max=3600, step=17),
                io.Autogrow.Input("prompts", template=io.Autogrow.TemplatePrefix(
                    input=io.Conditioning.Input("positive"), prefix="positive_", min=0, max=100),
                    optional=True),
                io.Int.Input("total_steps", default=8, min=1, max=10000, optional=True,
                             tooltip="Match both samplers' total steps. Allows a full sequential split (8+0 with the supplied PDD LoRA)."),
                MMH3CondSet.Input("cond_set", optional=True,
                                  tooltip="One conditioning per window from MMH3 Reference (Multi-Prompt), instead of the prompt sockets."),
                io.Latent.Input("latent", optional=True,
                                tooltip="The master to sample: a source encode, or a continuation master on resume. Sample this node's `latent` output, not the raw master."),
                io.Mask.Input("denoise_mask", optional=True,
                              tooltip="White regenerates, black keeps the source. Needs `latent`."),
                io.Mask.Input("audio_denoise_mask", optional=True,
                              tooltip="Same, for the soundtrack. Needs `latent`."),
                io.Combo.Input("denoise_mask_mode", options=MASK_MODES, default="max", optional=True,
                               tooltip="How pixel mask values are reduced onto a latent row."),
                io.Int.Input("accepted_prefix_frames", default=0, min=0, max=1000000, optional=True,
                             tooltip="Finished frames at the start of `latent` to preserve, including audio. On resume wire one chunk's frame count; 0 on fresh runs."),
                io.Int.Input("start_window", default=0, min=0, max=100000, optional=True,
                             tooltip="Global index of this run's first window, for per-window seeds. Wire the resume index shift."),
                io.Combo.Input("noise_mode", options=NOISE_MODES, default="global", optional=True,
                               tooltip="per_window draws seed + start_window + index per window, matching the single-node hybrid sampler; global slices one draw."),
            ],
            outputs=[io.Model.Output(display_name="sequential_model"),
                     io.Model.Output(display_name="joint_model"),
                     io.Conditioning.Output(display_name="positive"),
                     io.Int.Output(display_name="total_frames"),
                     io.Latent.Output(display_name="latent"),
                     io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, model, window_frames, overlap_frames, prompts=None, total_steps=8,
                cond_set=None, latent=None, denoise_mask=None, audio_denoise_mask=None,
                denoise_mask_mode="max", accepted_prefix_frames=0, start_window=0,
                noise_mode="global"):
        if "context_handler" in model.model_options:
            raise ValueError("Connect a model without another context-window adapter.")
        bound, count = bind_conditioning(cond_set, prompts)
        plan = plan_windows(window_frames, overlap_frames, count)
        run = HybridRun(plan, total_steps, noise_mode, start_window)
        prepared = prepare_source(run, latent, denoise_mask, audio_denoise_mask,
                                  denoise_mask_mode, accepted_prefix_frames)
        report = run_report(run, count, prepared)
        sequential = model.clone()
        add_outermost_wrapper(sequential, WrappersMP.OUTER_SAMPLE, "hybrid_sequential", SequentialWindows(run))
        joint = model.clone()
        joint.model_options["context_handler"] = JointWindows(plan)
        add_outermost_wrapper(joint, WrappersMP.OUTER_SAMPLE, "hybrid_joint", JointStage(run))
        joint.add_wrapper_with_key(WrappersMP.PREPARE_SAMPLING, "hybrid_joint_memory", prepare_joint)
        logging.info("[Hybrid Windows] %s", report)
        return io.NodeOutput(sequential, joint, bound, plan.total_frames, prepared, report)
