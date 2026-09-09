"""Hybrid windows through ordinary KSampler Advanced MODEL/LATENT connections."""

import logging
import math

import torch

import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.utils
from comfy.patcher_extension import WrappersMP
from comfy_api.latest import io

from .windows import (JointWindows, WINDOW_KEY, plan_windows,
                      select_conditioning, slice_av, window_options, write_new)


class SequentialWindows:
    def __init__(self, plan):
        self.plan = plan

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                 callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        if sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler:
            raise ValueError("Hybrid warmup currently supports plain Euler; select euler in KSampler Advanced.")
        if sampler.extra_options.get("s_churn", 0) or sampler.inpaint_options.get("random", False):
            raise ValueError("Hybrid warmup needs Euler without churn or random inpaint noise.")
        if denoise_mask is not None and torch.any(denoise_mask != 1):
            raise ValueError("This standalone workflow uses an empty AV latent. Source-mask preservation is not implemented.")
        if not 0 < float(sigmas[-1]) < float(sigmas[0]):
            raise ValueError("Warmup must stop before the final step with return_with_leftover_noise enabled.")
        if torch.count_nonzero(noise) == 0:
            raise ValueError("Enable add_noise on the first KSampler Advanced.")
        guider = executor.class_obj
        original_conds, original_options = guider.conds, guider.model_options
        shapes = latent_shapes
        spans = self.plan.spans(shapes)
        clean = [x.clone() for x in comfy.utils.unpack_latents(latent_image, shapes)]
        noise_parts = comfy.utils.unpack_latents(noise, shapes)
        output = [torch.empty_like(x, dtype=torch.float32) for x in clean]
        previous = (0, 0)
        steps = len(sigmas) - 1
        try:
            for index, span in enumerate(spans):
                comfy.model_management.throw_exception_if_processing_interrupted()
                source = [x.clone() for x in slice_av(clean, span)]
                mask = [torch.ones_like(x) for x in source]
                carry_v, carry_a = max(0, previous[0]-span[0]), max(0, previous[1]-span[2])
                mask[0][:, :, :carry_v] = 0
                mask[1][:, :, :, :carry_a] = 0
                source_packed, sub_shapes = comfy.utils.pack_latents(source)
                noise_packed, _ = comfy.utils.pack_latents(slice_av(noise_parts, span))
                # CFGGuider normally prepares masks before OUTER_SAMPLE. These
                # carry masks are created here, so honour that same device contract.
                mask_packed = (comfy.utils.pack_latents(mask)[0].to(guider.model_patcher.load_device)
                               if carry_v or carry_a else None)
                guider.conds = {key: [dict(c) for c in select_conditioning(group, index)]
                                for key, group in original_conds.items()}
                guider.model_options = window_options(original_options, span)
                last_prediction = None

                def capture(step, x0, x, total_steps):
                    nonlocal last_prediction
                    # Native process_latent_out also removes H3's carried-audio scale.
                    last_prediction = guider.inner_model.process_latent_out(x0).to(
                        device=latent_image.device, dtype=torch.float32).clone()

                logging.info("[Hybrid Windows] sequential window %d/%d, %d steps", index + 1, len(spans), steps)
                result = executor(noise_packed, source_packed, sampler, sigmas,
                                  mask_packed, capture, disable_pbar, seed, latent_shapes=sub_shapes)
                # Keep the native leftover-noise representation, owned by the first
                # window covering each position. No x0 substitution or fresh noise.
                write_new(output, comfy.utils.unpack_latents(result, sub_shapes), span, previous)
                prediction = comfy.utils.unpack_latents(last_prediction, sub_shapes)
                clean[0][:, :, span[0]:span[1]] = prediction[0].to(clean[0])
                clean[1][:, :, :, span[2]:span[3]] = prediction[1].to(clean[1])
                previous = (span[1], span[3])
                if callback is not None:
                    preview = comfy.utils.pack_latents(clean)[0]
                    callback((index+1)*steps-1, preview, preview, len(spans)*steps)
        finally:
            guider.conds, guider.model_options = original_conds, original_options
        return comfy.utils.pack_latents(output)[0]


def validate_joint(executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                   callback=None, disable_pbar=False, seed=None, latent_shapes=None):
    if torch.count_nonzero(noise):
        raise ValueError("Disable add_noise on the joint KSampler Advanced; connect the warmup LATENT output.")
    if denoise_mask is not None and torch.any(denoise_mask != 1):
        raise ValueError("The joint stage requires an unmasked warmup output.")
    if sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler:
        raise ValueError("Use plain euler in both KSampler Advanced nodes.")
    if float(sigmas[0]) >= 1:
        raise ValueError("Set joint start_at_step to the warmup end_at_step.")
    guider = executor.class_obj
    original = guider.model_options
    guider.model_options = {**original, "hybrid_latent_shapes": latent_shapes}
    try:
        return executor(noise, latent_image, sampler, sigmas, denoise_mask, callback,
                        disable_pbar, seed, latent_shapes=latent_shapes)
    finally:
        guider.model_options = original


def prepare_joint(executor, model, noise_shape, conds, **kwargs):
    options = kwargs["model_options"]
    shapes = options["hybrid_latent_shapes"]
    plan = options["context_handler"].plan
    spans = plan.spans(shapes)
    # Budget the largest AV window instead of a fictitious full-timeline forward.
    video_size = shapes[0][1] * plan.length * math.prod(shapes[0][3:])
    audio_size = math.prod(shapes[1][1:3]) * max(a1-a0 for _, _, a0, a1 in spans)
    return executor(model, [noise_shape[0], 1, video_size+audio_size], conds, **kwargs)


class H3HybridWindows(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3HybridWindows", display_name="H3 Hybrid Windows",
            category="sampling/hybrid", is_experimental=True,
            description="Sequential overlap carry followed by joint prediction fusion through two native KSampler Advanced nodes.",
            inputs=[
                io.Model.Input("model"),
                io.Int.Input("window_frames", default=243, min=22, max=3600, step=17),
                io.Int.Input("overlap_frames", default=39, min=5, max=3600, step=17),
                io.Autogrow.Input("prompts", template=io.Autogrow.TemplatePrefix(
                    input=io.Conditioning.Input("positive"), prefix="positive_", min=1, max=100)),
            ],
            outputs=[io.Model.Output(display_name="sequential_model"),
                     io.Model.Output(display_name="joint_model"),
                     io.Conditioning.Output(display_name="positive"),
                     io.Int.Output(display_name="total_frames")],
        )

    @classmethod
    def execute(cls, model, window_frames, overlap_frames, prompts):
        if "context_handler" in model.model_options:
            raise ValueError("Connect a model without another context-window adapter.")
        groups = [prompts[name] for name in sorted(prompts, key=lambda name: int(name.rsplit("_", 1)[1]))]
        plan = plan_windows(window_frames, overlap_frames, len(groups))
        bound = []
        for index, group in enumerate(groups):
            if not group:
                raise ValueError(f"Connect conditioning for prompt {index + 1}.")
            for tokens, metadata in group:
                # Native image guides use positions within this conditioning's
                # window. Window selection keeps them scoped in both stages.
                bound.append([tokens, {**metadata, WINDOW_KEY: index}])
        sequential = model.clone()
        sequential.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "hybrid_sequential", SequentialWindows(plan))
        joint = model.clone()
        joint.model_options["context_handler"] = JointWindows(plan)
        joint.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "hybrid_joint", validate_joint)
        joint.add_wrapper_with_key(WrappersMP.PREPARE_SAMPLING, "hybrid_joint_memory", prepare_joint)
        return io.NodeOutput(sequential, joint, bound, plan.total_frames)
