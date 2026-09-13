"""Solver segment helpers shared by the single-node sampler and the native chain.

Both paths hand the solver a state that ordinary sampling would have produced
itself, so the pieces live here rather than in either caller. Core imports only:
`nodes.py` must stay importable without MMH3Tools.
"""

import torch

from comfy.samplers import KSAMPLER


class FixedNoise:
    def __init__(self, samples, seed):
        self.samples = samples
        self.seed = seed

    def generate_noise(self, latent):
        return self.samples


class SolverSegment(KSAMPLER):
    """Keep raw x_sigma before KSAMPLER applies inverse_noise_scaling.

    On resume, only the initial solver state is replaced. The native inpaint
    wrapper still receives the original source latent, mask and fixed noise.

    ANY solver is accepted. `x` is the whole state for memoryless solvers
    (euler, heun, dpm_2), so those resume exactly. Multistep solvers (dpmpp_2m,
    res_multistep) also carry `old_denoised`, which is NOT captured: a resumed
    run restarts its history and spends one first-order step at the boundary --
    the same thing it does at step 0 of any ordinary run. Stochastic solvers
    (ancestral, SDE, s_churn) resume from a valid state, but a run is not
    reproducible across the switch because each phase draws its own noise.

    `inpaint_options["random"]` stays rejected, and that is not a solver
    preference: it replaces the sampler's noise with a fresh draw, discarding
    the per-window noise this design assembles and re-injects into pinned rows
    on both sides of the switch. That noise IS the handoff.
    """

    def __init__(self, sampler, initial_state=None, capture=False):
        if sampler.inpaint_options.get("random", False):
            raise ValueError(
                "Hybrid sampling cannot use random inpaint noise: pinned rows must be re-noised "
                "with the same per-window noise on both sides of the switch.")
        self.initial_state = initial_state
        self.solver = sampler.sampler_function
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

        result = self.solver(model, x, sigmas, extra_args, record, disable, **options)
        if self.capture:
            self.final_state = result.detach().to(device="cpu", dtype=torch.float32).clone()
            self.final_prediction = last_prediction.detach().to(device="cpu", dtype=torch.float32).clone()
        return result


# The name this class carried while it accepted only Euler. Both spellings stay
# importable so the native chain and the single-node sampler keep working.
EulerSegment = SolverSegment
