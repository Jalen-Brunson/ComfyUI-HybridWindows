# Looping Sampler vs Hybrid on stream_00170 - how the example was made

Two ComfyUI API prompts built by `build_prompts.py` (2026-09-10) share everything but the sampler:

- source `ComfyUI/input/stream_00170.mp4` (the MMH3Tools example clip, 768x576, 24 fps, 2895 frames) decoded once
  by core LoadVideo -> Video Slice -> GetVideoComponents -> ImageAddNoise 0.1 (seed 123) and used as `<Video 1>`;
- one reference picture of the subject as `<Picture 1>` (not included in this repository);
- `minimax_h3_ref2va_int8_convrot` -> LoraLoaderModelOnly `MiniMax-H3-Ref2VA-Acc-8Step_comfy` 1.0 -> sigma shift 12/3,
  qwen3vl_32b int8 CLIP, int8 video VAE, fp32 audio VAE, plain SDPA attention;
- 768x576, 14 windows of 243 frames / 39-frame overlap (window i starts at frame 204*i), one prompt for every window,
  seed 123, euler / simple, 8 steps, CFG 1, generated audio, no post-processing.

`hybrid_native.api.json`: 14 native MiniMaxH3ReferenceToVideo conditionings (ref image + the window's 243 noised
source frames) -> H3HybridWindows -> KSamplerAdvanced steps 0-6 (leftover noise) -> KSamplerAdvanced steps 6-8 (joint).
`mmh3_looping.api.json`: MMH3ReferenceMultiPrompt (window_ref_video, chunk 243 / overlap 39, length 2895) ->
MMH3LoopingSampler (carry mask, overlap strength 1.0 video / 0.9 audio, 8 steps, all optional features off).

Pure sampling time = the sum of the sampler progress bars in the server log (model loads, reference encoding and
VAE decoding excluded): 43:14 hybrid (14 x 6-step warm-up bars + one 2-step joint bar) vs 43:17 looping (14 x 8-step bars).
Likeness = insightface buffalo_l ArcFace cosine of the largest face per sampled frame against the reference picture.
The per-window chunk-quality panels come from the `chunk_quality.py` readout of the vlm_video_prompt pack (not part
of this repository). `color_correct.sh` is the chroma-only grade applied identically to both renders.
