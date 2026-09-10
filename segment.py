"""Euler segment helpers shared by the single-node sampler and the native chain.

Both paths hand the solver a state that ordinary sampling would have produced
itself, so the pieces live here rather than in either caller. Core imports only:
`nodes.py` must stay importable without MMH3Tools.
"""

import torch

import comfy.k_diffusion.sampling as k_diffusion_sampling
from comfy.samplers import KSAMPLER


class FixedNoise:
    def __init__(self, samples, seed):
        self.samples = samples
        self.seed = seed

    def generate_noise(self, latent):
        return self.samples


class EulerSegment(KSAMPLER):
    """Keep raw x_sigma before KSAMPLER applies inverse_noise_scaling.

    On resume, only the initial solver state is replaced. The native inpaint
    wrapper still receives the original source latent, mask and fixed noise.
    """

    def __init__(self, sampler, initial_state=None, capture=False):
        if sampler.sampler_function is not k_diffusion_sampling.sample_euler:
            raise ValueError("Hybrid sampling requires plain Euler; other solvers need separate state handling.")
        if sampler.extra_options.get("s_churn", 0) != 0 or sampler.inpaint_options.get("random", False):
            raise ValueError("Hybrid sampling requires Euler without churn or random inpaint noise.")
        self.initial_state = initial_state
        self.euler = sampler.sampler_function
        self.capture = capture
        self.final_state = None
        self.final_prediction = None
        super().__init__(self._run, dict(sampler.extra_options), dict(sampler.inpaint_options))

    def _run(self, model, x, sigmas, extra_args=None, callback=None, disable=None, **options):
        if self.initial_state is not None:
            if self.initial_state.shape != x.shape:
                raise ValueError("Hybrid resume state does not match the packed AV latent shape.")
            x = self.initial_state.to(device=x.device, dtype=x.dtype).clone()
        last_prediction = None

        def record(data):
            nonlocal last_prediction
            last_prediction = data["denoised"]
            if callback is not None:
                callback(data)

        result = self.euler(model, x, sigmas, extra_args, record, disable, **options)
        if self.capture:
            self.final_state = result.detach().to(device="cpu", dtype=torch.float32).clone()
            self.final_prediction = last_prediction.detach().to(device="cpu", dtype=torch.float32).clone()
        return result
