"""Use ComfyUI's Fun Control patch at each window's absolute source offset."""

from dataclasses import dataclass, field

import torch

from comfy.patcher_extension import WrappersMP
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3FunControlPatch

from .windows import OFFSET_KEY

CACHE_KEY = "hybrid_control_run"


@dataclass
class ControlRun:
    latents: dict = field(default_factory=dict)


def control_run(executor, *args, **kwargs):
    guider = executor.class_obj
    original = guider.model_options
    state = ControlRun()
    guider.model_options = {**original, "transformer_options": {
        **original.get("transformer_options", {}), CACHE_KEY: state,
    }}
    try:
        return executor(*args, **kwargs)
    finally:
        state.latents.clear()
        guider.model_options = original


class WindowControl(MiniMaxH3FunControlPatch):
    def diffusion_model_wrapper(self, executor, x, timestep, context, transformer_options={}, **kwargs):
        offset = int(transformer_options.get(OFFSET_KEY, 0))
        state = transformer_options[CACHE_KEY]
        key = (id(self), offset, tuple(x[0].shape))
        sigmas = transformer_options.get("sigmas")
        sigma = float(sigmas[0]) if sigmas is not None else float(timestep.flatten()[0])/1000
        try:
            if self.sigma_end <= sigma <= self.sigma_start:
                if key not in state.latents:
                    original = self.control_video
                    self.control_video = original[min(offset, len(original)-1):]
                    try:
                        super().prepare_control_latent(x[0].shape)
                        state.latents[key] = self.control_latent.to(device="cpu", dtype=torch.float32)
                    finally:
                        self.control_video = original
                self.control_latent = state.latents[key]
                self.control_latent_shape = tuple(x[0].shape)
            return super().diffusion_model_wrapper(executor, x, timestep, context, transformer_options, **kwargs)
        finally:
            # The cache belongs to this sampling call, not to the MODEL output.
            self.control_latent = None
            self.control_latent_shape = None


class H3HybridControlNet(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3HybridControlNet", display_name="H3 Window ControlNet",
            category="sampling/hybrid",
            description="Native Fun ControlNet with source-frame offsets for hybrid windows. Add control noise upstream with ComfyUI's Add Noise to Image.",
            inputs=[io.Model.Input("model"), io.ModelPatch.Input("model_patch"),
                    io.Vae.Input("vae"), io.Image.Input("control_video"),
                    io.Float.Input("strength", default=0.5, min=0, max=10, step=0.01),
                    io.Float.Input("start_percent", default=0, min=0, max=1, step=0.01),
                    io.Float.Input("end_percent", default=1, min=0, max=1, step=0.01)],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(cls, model, model_patch, vae, control_video, strength, start_percent=0, end_percent=1):
        if strength == 0:
            return io.NodeOutput(model)
        patched = model.clone()
        sampling = model.get_model_object("model_sampling")
        control = WindowControl(model_patch, vae, control_video[..., :3].movedim(-1, 1),
                                None, None, strength, float(sampling.percent_to_sigma(start_percent)),
                                float(sampling.percent_to_sigma(end_percent)))
        control.register(patched)
        patched.add_wrapper_with_key(WrappersMP.OUTER_SAMPLE, "hybrid_control_run", control_run)
        return io.NodeOutput(patched)
