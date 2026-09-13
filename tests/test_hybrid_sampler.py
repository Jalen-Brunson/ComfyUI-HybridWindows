"""State continuation and ownership checks against Comfy's actual Euler/inpaint path."""

import sys
import os
import importlib
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT.parents[1]), str(ROOT), os.environ.get("MMH3TOOLS_TEST_ROOT", str(ROOT.parent / "ComfyUI-MMH3Tools"))]
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import torch
import comfy.utils
from comfy.model_sampling import CONST
from comfy.nested_tensor import NestedTensor
from comfy.samplers import ksampler
from comfy_api.latest import io
from mmh3tools.common import frame_at_latent, frames_to_audio_t
import nodes
from mmh3tools.nodes_looping_sampler import MMH3LoopingSampler
nodes.NODE_CLASS_MAPPINGS["MMH3LoopingSampler"] = MMH3LoopingSampler
# Package import also verifies that startup needs no registered joint sampler.
spec = importlib.util.spec_from_file_location("hybridwindows_test", ROOT / "__init__.py")
package = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = package
spec.loader.exec_module(package)
from hybridwindows_test.sampler import (
    EulerSegment, FixedNoise, MMH3HybridWindowSampler, SolverSegment, _write_new,
)

torch.set_num_threads(2)
SIGMAS = torch.tensor([1., .988235294, .972972973, .952380952,
                       .923076923, .87804878, .8, .631578947, 0.])


class Model:
    def __init__(self):
        sampling = CONST()
        sampling.sigma_max = 1.
        self.inner_model = SimpleNamespace(model_sampling=sampling, scale_latent_inpaint=self.inpaint)
        # res_multistep and friends reach for get_model_object('model_sampling').
        self.model_patcher = SimpleNamespace(
            model=self, get_model_object=lambda name: getattr(self.inner_model, name, None))
        self.cfg = 1.
        self.seen = []

    def inpaint(self, x, sigma, noise, latent_image, denoise_mask):
        return .999 * latent_image + .001 * noise

    def __call__(self, x, sigma, **kwargs):
        self.seen.append((sigma.clone(), x.clone()))
        return .27 * x.tanh() + .13 * x.mean() + .31 * sigma.reshape(-1, 1, 1)


def run(sampler, model, sigmas, noise, source, mask=None):
    return sampler.sample(model, sigmas, {"model_options": {}, "seed": 123},
                          None, noise, source, mask, disable_pbar=True)


class HybridTests(unittest.TestCase):
    def test_resume_matches_uninterrupted_at_every_supported_split(self):
        generator = torch.Generator().manual_seed(456)
        noise = torch.randn(1, 1, 240, generator=generator)
        source = torch.randn(1, 1, 240, generator=generator)
        for with_mask in (False, True):
            mask = None
            if with_mask:
                mask = torch.ones_like(source)
                mask[..., :40] = 0
                mask[..., 80:120] = 0
            reference_model = Model()
            reference = run(ksampler("euler"), reference_model, SIGMAS, noise, source, mask)
            for split in range(1, 8):
                model = Model()
                first = EulerSegment(ksampler("euler"), capture=True)
                exposed = run(first, model, SIGMAS[:split + 1], noise, source, mask)
                self.assertFalse(torch.equal(exposed, first.final_state))
                self.assertFalse(torch.equal(first.final_prediction, first.final_state))
                resume = EulerSegment(ksampler("euler"), first.final_state)
                output = run(resume, model, SIGMAS[split:], noise, source, mask)
                torch.testing.assert_close(output, reference, rtol=0, atol=0)
                for (s1, x1), (s2, x2) in zip(model.seen, reference_model.seen, strict=True):
                    torch.testing.assert_close(s1, s2, rtol=0, atol=0)
                    torch.testing.assert_close(x1, x2, rtol=0, atol=0)
                if mask is not None:
                    torch.testing.assert_close(output[mask == 0], source[mask == 0], rtol=1e-6, atol=3e-7)

    def test_memoryless_solvers_resume_exactly_and_multistep_loses_history(self):
        """The capture is `x` alone, so exactness depends on the solver's state.

        Memoryless solvers (one step reads only the previous x) resume bit-exact.
        Multistep solvers also carry `old_denoised`, which is not captured, so the
        first step after the switch is first-order -- the same thing they do at
        step 0 of any ordinary run.
        """
        generator = torch.Generator().manual_seed(456)
        noise = torch.randn(1, 1, 240, generator=generator)
        source = torch.randn(1, 1, 240, generator=generator)
        split = 6
        for name in ("heun", "dpm_2"):
            reference = run(ksampler(name), Model(), SIGMAS, noise, source)
            first = SolverSegment(ksampler(name), capture=True)
            run(first, Model(), SIGMAS[:split + 1], noise, source)
            resumed = run(SolverSegment(ksampler(name), first.final_state),
                          Model(), SIGMAS[split:], noise, source)
            torch.testing.assert_close(resumed, reference, rtol=0, atol=0,
                                       msg=f"{name} must resume exactly from x alone")
        for name in ("dpmpp_2m", "res_multistep"):
            reference = run(ksampler(name), Model(), SIGMAS, noise, source)
            first = SolverSegment(ksampler(name), capture=True)
            run(first, Model(), SIGMAS[:split + 1], noise, source)
            resumed = run(SolverSegment(ksampler(name), first.final_state),
                          Model(), SIGMAS[split:], noise, source)
            self.assertFalse(torch.equal(resumed, reference),
                             f"{name} carries history the capture cannot restore")

    def test_wrong_resume_from_clean_prediction_is_detectable(self):
        source = torch.full((1, 1, 40), .2)
        noise = torch.linspace(-1, 1, 40).reshape_as(source)
        first = EulerSegment(ksampler("euler"), capture=True)
        run(first, Model(), SIGMAS[:7], noise, source)
        correct = run(EulerSegment(ksampler("euler"), first.final_state), Model(), SIGMAS[6:], noise, source)
        wrong = run(EulerSegment(ksampler("euler"), first.final_prediction), Model(), SIGMAS[6:], noise, source)
        self.assertGreater(float((correct - wrong).abs().max()), .01)

    def test_first_owner_survives_large_tail_overlap_for_state_and_noise(self):
        destination = (torch.full((1, 24, 62, 2, 2), -1.), torch.full((1, 32, 2, 348), -1.))
        spans = [(0, 37, 0, 207), (25, 62, 142, 348), (25, 62, 142, 348)]
        previous_v = previous_a = 0
        for value, (v0, v1, a0, a1) in enumerate(spans):
            source = (torch.full((1, 24, v1-v0, 2, 2), float(value)),
                      torch.full((1, 32, 2, a1-a0), float(value)))
            _write_new(destination, source, v0, v1, a0, a1, previous_v, previous_a)
            previous_v, previous_a = v1, a1
        v, a = destination
        self.assertTrue(bool((v[:, :, :37] == 0).all()))
        self.assertTrue(bool((v[:, :, 37:] == 1).all()))
        self.assertTrue(bool((a[..., :207] == 0).all()))
        self.assertTrue(bool((a[..., 207:] == 1).all()))

    def test_accepts_any_solver_but_not_random_inpaint_noise(self):
        # Other solvers are allowed: the joint stage starts with no history,
        # which is any multistep solver's ordinary step-0 condition.
        for name in ("euler", "heun", "dpm_2", "dpmpp_2m", "res_multistep", "euler_ancestral"):
            segment = SolverSegment(ksampler(name))
            self.assertIs(segment.solver, ksampler(name).sampler_function)
        SolverSegment(ksampler("euler", {"s_churn": 1.}))
        # Random inpaint noise is not a solver preference: it discards the
        # per-window noise that re-injects pinned rows on both sides of the switch.
        with self.assertRaisesRegex(ValueError, "random inpaint noise"):
            SolverSegment(ksampler("euler", inpaint_options={"random": True}))

    def test_sampler_module_reload_during_custom_node_startup(self):
        import comfy.k_diffusion.sampling as sampling
        old = sampling.sample_euler
        importlib.reload(sampling)  # RES4LYF does this after other packs import.
        self.assertIsNot(old, sampling.sample_euler)
        segment = EulerSegment(ksampler("euler"))
        self.assertIs(segment.solver, sampling.sample_euler)

    def test_node_uses_raw_state_releases_only_generated_carry_and_aligns_controls(self):
        import hybridwindows_test.sampler as hybrid
        video = torch.zeros(1, 24, 62, 2, 2)
        audio = torch.full((1, 32, 2, frames_to_audio_t(209)), .3)
        mv = torch.ones(1, 1, 62, 2, 2)
        mv[..., 0, 0] = 0
        ma = torch.zeros(1, 1, 2, audio.shape[-1])
        latent = {"samples": NestedTensor([video, audio]), "noise_mask": NestedTensor([mv, ma])}
        conds = {"conds": [[[torch.zeros(1), {"marker": i}]] for i in range(2)]}
        model = SimpleNamespace(model_options={}, model=SimpleNamespace(process_latent_out=lambda x: x))
        model.clone = lambda: SimpleNamespace(model_options={})
        calls = []

        class Guider:
            def __init__(self, m):
                self.model_options = m.model_options
                self.original_conds = {}
            def set_conds(self, cond):
                self.original_conds["positive"] = cond

        class Noise:
            seed = 100
            def generate_noise(self, lat):
                return NestedTensor([torch.full_like(x, float(self.seed)) for x in lat["samples"].unbind()])

        def sample(noise, guider, sampler, sigmas, chunk):
            parts = chunk["samples"].unbind()
            call = len(calls)
            calls.append((noise, guider, sampler, sigmas, chunk))
            if call < 2:
                self.assertEqual(noise.seed, 100 + call)
                self.assertEqual(guider.model_options["transformer_options"]["mmh3_control_frame0"],
                                 frame_at_latent(25 * call))
                self.assertEqual(guider.original_conds["positive"][0][1]["marker"], call)
                cv, ca = chunk["noise_mask"].unbind()
                self.assertTrue(bool((ca == 0).all()))
                if call == 1:
                    self.assertTrue(bool((cv[:, :, :12] == 0).all()))
                    self.assertTrue(bool((parts[0][:, :, :12] == 10).all()))
                raw = [torch.full_like(x, 20. + call) for x in parts]
                pred = [torch.full_like(x, 10. + call) for x in parts]
                sampler.final_state = comfy.utils.pack_latents(raw)[0]
                sampler.final_prediction = comfy.utils.pack_latents(pred)[0]
            else:
                v, a = comfy.utils.unpack_latents(sampler.initial_state, [video.shape, audio.shape])
                self.assertTrue(bool((v[:, :, :37] == 20).all()))
                self.assertTrue(bool((v[:, :, 37:] == 21).all()))
                torch.testing.assert_close(chunk["noise_mask"].unbind()[0], mv, rtol=0, atol=0)
                torch.testing.assert_close(chunk["noise_mask"].unbind()[1], ma, rtol=0, atol=0)
                torch.testing.assert_close(parts[0], video, rtol=0, atol=0)
                torch.testing.assert_close(parts[1], audio, rtol=0, atol=0)
                nv, na = noise.samples.unbind()
                self.assertTrue(bool((nv[:, :, :37] == 100).all()))
                self.assertTrue(bool((nv[:, :, 37:] == 101).all()))
                self.assertEqual(float(sigmas[0]), float(SIGMAS[6]))
            return io.NodeOutput(chunk, chunk)

        with patch.object(hybrid, "Guider_Basic", Guider), \
             patch.object(hybrid, "create_prepare_sampling_wrapper", lambda m: None), \
             patch.object(hybrid.SamplerCustomAdvanced, "execute", side_effect=sample):
            MMH3HybridWindowSampler.execute(model, Noise(), ksampler("euler"), SIGMAS,
                                            conds, latent, 124, 39, 6)
        self.assertEqual(len(calls), 3)
        self.assertEqual(model.model_options, {})
        torch.testing.assert_close(latent["samples"].unbind()[0], video, rtol=0, atol=0)

    def test_registration(self):
        from mmh3tools import NODES
        self.assertNotIn(MMH3HybridWindowSampler, NODES)
        self.assertEqual(MMH3HybridWindowSampler.GET_SCHEMA().node_id, "MMH3HybridWindowSampler")

    def test_finished_prefix_survives_both_phases_and_global_seeds(self):
        import hybridwindows_test.sampler as hybrid
        video = torch.arange(24*77*4).reshape(1,24,77,2,2).float()/1000
        audio = torch.linspace(-.2,.2,32*2*frames_to_audio_t(260)).reshape(1,32,2,-1)
        vm = torch.ones(1,1,77,2,2)
        vm[...,0,0] = 0
        ma = torch.ones(1,1,2,audio.shape[-1])
        latent = {"samples": NestedTensor([video,audio]), "noise_mask": NestedTensor([vm,ma])}
        original_v, original_a = video.clone(), audio.clone()
        original_mask = vm.clone()
        conds = {"conds": [[[torch.zeros(1), {"marker": i}]] for i in range(3)]}
        model = SimpleNamespace(model_options={}, model=SimpleNamespace(process_latent_out=lambda x:x))
        model.clone = lambda: SimpleNamespace(model_options={})
        calls = []

        class Guider:
            def __init__(self,m):
                self.model_options=m.model_options
                self.original_conds={}
            def set_conds(self,c):self.original_conds['positive']=c

        class Noise:
            seed=100
            def generate_noise(self,lat):
                return NestedTensor([torch.full_like(x,float(self.seed)) for x in lat['samples'].unbind()])

        from mmh3tools.nodes_windows import _audio_index_at
        accepted_a = _audio_index_at(37,77,audio.shape[-1])

        def sample(noise,guider,sampler,sigmas,chunk):
            call=len(calls);calls.append(call)
            parts=chunk['samples'].unbind()
            cv,ca=chunk['noise_mask'].unbind()
            if call<2:
                # Window 0 is accepted and does not call the model. The final
                # window is clamped back to latent 40, enlarging its overlap.
                i=call+1;start=[25,40][call]
                self.assertEqual(noise.seed,103+i)
                self.assertEqual(guider.original_conds['positive'][0][1]['marker'],i)
                self.assertEqual(guider.model_options['transformer_options']['mmh3_control_frame0'],frame_at_latent(start))
                if call==0:
                    torch.testing.assert_close(parts[0][:,:,:12],video[:,:,25:37],rtol=0,atol=0)
                    self.assertTrue(bool((cv[:,:,:12]==0).all()))
                raw=[torch.full_like(x,20.+i) for x in parts]
                pred=[torch.full_like(x,10.+i) for x in parts]
                sampler.final_state=comfy.utils.pack_latents(raw)[0]
                sampler.final_prediction=comfy.utils.pack_latents(pred)[0]
            else:
                self.assertTrue(bool((cv[:,:,:37]==0).all()))
                self.assertTrue(bool((ca[...,:accepted_a]==0).all()))
                self.assertTrue(bool((cv[:,:,37:,1,1]==1).all()))
                self.assertTrue(bool((cv[...,0,0]==0).all()))
                self.assertTrue(bool((ca[...,accepted_a:]==1).all()))
                torch.testing.assert_close(parts[0][:,:,:37],video[:,:,:37],rtol=0,atol=0)
                rv,ra=comfy.utils.unpack_latents(sampler.initial_state,[video.shape,audio.shape])
                self.assertTrue(bool(torch.isfinite(rv).all() and torch.isfinite(ra).all()))
                self.assertTrue(bool((rv[:,:,37:62]==21).all()))
                self.assertTrue(bool((rv[:,:,62:]==22).all()))
                nv,na=noise.samples.unbind()
                self.assertTrue(bool((nv[:,:,:37]==103).all()))
                self.assertTrue(bool((nv[:,:,37:62]==104).all()))
                self.assertTrue(bool((nv[:,:,62:]==105).all()))
            bad={**chunk,'samples':NestedTensor([torch.full_like(x,-999.) for x in parts])}
            return io.NodeOutput(bad,bad)

        with patch.object(hybrid,'Guider_Basic',Guider), \
             patch.object(hybrid,'create_prepare_sampling_wrapper',lambda m:None), \
             patch.object(hybrid.SamplerCustomAdvanced,'execute',side_effect=sample):
            out=MMH3HybridWindowSampler.execute(model,Noise(),ksampler('euler'),SIGMAS,
                conds,latent,124,39,6,accepted_prefix_frames=124,start_window=3)[0]
        ov,oa=out['samples'].unbind()
        self.assertEqual(len(calls),3)
        torch.testing.assert_close(ov[:,:,:37],video[:,:,:37],rtol=0,atol=0)
        torch.testing.assert_close(oa[...,:accepted_a],audio[...,:accepted_a],rtol=0,atol=0)
        self.assertTrue(bool((ov[:,:,37:]==-999).all()))
        torch.testing.assert_close(video,original_v,rtol=0,atol=0)
        torch.testing.assert_close(audio,original_a,rtol=0,atol=0)
        torch.testing.assert_close(vm,original_mask,rtol=0,atol=0)

    def test_fully_accepted_input_does_not_sample(self):
        import hybridwindows_test.sampler as hybrid
        v=torch.randn(1,24,37,2,2);a=torch.randn(1,32,2,frames_to_audio_t(124))
        latent={'samples':NestedTensor([v,a])}
        model=SimpleNamespace(model_options={})
        conds={'conds':[[[torch.zeros(1),{}]]]}
        with patch.object(hybrid.SamplerCustomAdvanced,'execute',side_effect=AssertionError('sampled accepted input')):
            out=MMH3HybridWindowSampler.execute(model,None,ksampler('euler'),SIGMAS,
                conds,latent,124,39,6,accepted_prefix_frames=124)[0]
        for expected,actual in zip((v,a),out['samples'].unbind()):
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
        for accepted in (-1,6,125):
            with self.assertRaisesRegex(ValueError,'accepted_prefix_frames'):
                MMH3HybridWindowSampler.execute(model,None,ksampler('euler'),SIGMAS,
                    conds,latent,124,39,6,accepted_prefix_frames=accepted)


if __name__ == "__main__":
    unittest.main(argv=[sys.argv[0]], verbosity=2)
