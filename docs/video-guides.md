# Aligned video and audio guides

Connect source frames to `guide_video` and the H3 video VAE to `guide_vae` on
**H3 Hybrid Windows**. Both native samplers use its usual `positive` output.
The sampler wrappers slice the source using the actual window plan and encode
each window with native **Add Guide for MiniMax H3**, at local frame zero.

Guides are encoded once per window per sampling stage, before model loading.
Both hybrid upscalers retain the guide inputs: joint finishing re-encodes the
source at the actual high-resolution latent size. Encoded guides are scoped
to that sampling call; input conditioning and existing still guides are preserved.
Guide preparation runs during sampling, so changing the guide re-runs its VAE
encoding. There is no cross-run guide cache.

The standalone **H3 Hybrid Window Sampler** exposes the same inputs and shares
its encoded guides between the sequential and joint phases at its fixed resolution.
It also preserves guides already present in each window's conditioning set.

## Prompt counts

When the input latent needs more windows than the supplied prompts, both hybrid
samplers repeat the final conditioning for the remaining windows. For example,
`A | B` over four windows becomes `A | B | B | B`. A single prompt covers the
entire input. Guides still use each window's own source frames.

On **H3 Hybrid Windows**, connect the source to `latent` so its length determines
the required windows. Without that input, the prompt count still sets the output
length. Empty conditioning and surplus prompts remain errors.

## Timing and audio

- Supply frames at 24 fps, already trimmed to the current run, **including its
  accepted resume context**. `start_window` does not apply a second source offset.
- Window boundaries, overlap, and the last window's slide follow the sampling
  plan. The source must cover every planned window.
- Optionally connect an aligned soundtrack to `guide_audio`, with
  `guide_audio_vae`. Audio-only guides also work.
- Audio guides condition generation. Existing audio denoise masks control
  whether output audio is preserved.
- The log reports each guide's source frame range and sampling resolution.
  Guides use the native node's center-crop resizing and model conditioning;
  they do not paste source pixels into the output.

## Convert an existing V2V guide workflow

The converter replaces matching `H3CharacterSwapGuides` adapters with one set
of inputs on Hybrid Windows. It removes their unused setup and report nodes,
reconnects both Basic Guiders, and keeps the prompt, model, sampling, source,
and resume paths. Output/resume names get a `_hybrid_guides` variant.

```bash
python build_integrated_guide_workflow.py "V2V - Add Guide.json" "V2V - Hybrid Guides.json"
```

The destination must be a new file. The source workflow is preserved.

`python tests/test_guides.py` exercises native sampling with lightweight model
and VAE fixtures: window selection, short final windows, resume offsets, audio,
both upscalers, conditioning isolation, native Add Guide parity, and standalone
sampler parity. These checks do not measure generation quality.
