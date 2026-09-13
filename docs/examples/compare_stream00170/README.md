# Looping, Hybrid full and Hybrid context 5 on stream_00170

The [comparison videos and previews](../../../README.md#example-looping-sampler-vs-hybrid-on-a-two-minute-clip)
append the September 13 context-5 run to the original September 10 recordings.
The original videos were reused. All three use:

- `ComfyUI/input/stream_00170.mp4`: 768×576, 24 fps, 2895 frames / 120.625 seconds;
- `hybrid_man_reference.png` as `<Picture 1>` (the source and reference are not included here);
- Ref2VA int8 → `minimax/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors` at 1.0 → video/audio sigma shifts 12/3,
  qwen3vl_32b int8 CLIP, int8 video VAE, fp32 audio VAE and plain SDPA attention;
- 14 windows of 243 frames with 39-frame overlap, seed 123, Euler / simple, 8 steps, CFG 1,
  one repeated prompt and generated audio;
- core LoadVideo → Video Slice → GetVideoComponents → ImageAddNoise 0.1, seed 123,
  supplying the motion-reference video.

The runnable hybrid graph now selects **`overlap_pin = tail_custom`, `context_frames = 5`**.
Only those two inputs and the output filename prefix changed from the archived hybrid
prompt. The 39-frame overlap remains present; five frames controls pinned video carry
during sequential warmup. The run log reported an effective five frames at every
incoming overlap.

## Graphs and provenance

[`hybrid_native.api.json`](hybrid_native.api.json) is the actual context-5 prompt:
14 native MiniMaxH3ReferenceToVideo conditionings → H3HybridWindows → KSamplerAdvanced
steps 0–6 with leftover noise → KSamplerAdvanced steps 6–8 for the joint finish.
[`build_prompts.py`](build_prompts.py) reproduces this prompt and the Looping baseline.
Run it from any directory; it writes the two API JSON files beside the script.

[`mmh3_looping.api.json`](mmh3_looping.api.json) retains the original baseline:
MMH3ReferenceMultiPrompt → MMH3LoopingSampler, 8 sequential steps, carry mask,
video/audio overlap strengths 1.0/0.9 and optional features off. This separate sampler
has no Hybrid Windows `context_frames` control.

The full-context recording used the
[archived hybrid prompt](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/blob/d6d5bb2d271fbae2c9ac312b0b9f35724b773892/docs/examples/compare_stream00170/hybrid_native.api.json).
It omitted the new pin controls, selecting full pinning. The baselines ran on an H200;
the new run used the installed implementation and an H100 80GB. Implementation hashes
and the ComfyUI commit for the new run are recorded in [`results.json`](results.json).
These are one-clip, one-seed measurements, with no claim of a hardware-matched speed test.

## Measurements

Pure sampling time sums the sampler progress bars, excluding model loading, reference
encoding and VAE decoding: **43:17 Looping**, **43:14 Hybrid full**, **45:00 Hybrid context 5**.
The hybrids have fourteen 6-step warmup bars and one 2-step joint bar; Looping has fourteen
8-step bars. Each totals 112 model evaluations across windows.

Likeness uses InsightFace buffalo_l / ArcFace cosine against the reference picture:
40 evenly spaced whole-clip frames, largest face, any gender, minimum face height 40 px,
CUDA, and 24 source-performer reference frames for the separate source-similarity readout.
Per-window likeness is the median of six samples at zero-based frame
`window * 204 + int((sample + 0.5) * 243 / 6)`. The original per-window baseline measurements
were retained; whole-clip baselines were rechecked with the original 40-frame setting.

Quality panels use the original `compare_n.py` layout and `vlm_video_prompt/chunk_quality.py`
readout, with unchanged metric-module SHA-256
`4b672839c2003acf10c5358035f26a637caa895fc0fea0d6526d6139a07cfade`.
These analysis tools are not part of this repository. Window 1 is each render's own anchor,
and ratios account for the source trajectory. They measure drift within a render, not
absolute quality across different first-window anchors. Complete per-window verdicts,
ratios, likeness and thresholds are in [`results.json`](results.json).

Context 5 stayed **OK in all 14 windows**. Its whole-clip likeness median/p10 was
**0.743/0.714**; window-14 fine texture was **1.29×**, contrast **1.09×**, saturation
**1.02×**, and mottle **1.08×**. See the main README table for all three arms.

## Colour and comparison layouts

[`color_correct.sh`](color_correct.sh) records the original chroma-only grade. The same
settings were applied to the new render: reference seconds 1–8, chroma 0.85, gain floor
0.9, maximum chroma shift 0.025, and 2-second smoothing and ramp. Brightness, contrast,
texture and audio are unchanged.

The six release comparisons retain their original filenames and now show all three
renders: clean, per-window metrics, and Original-first, each raw and colour corrected.
Audio tracks 1–3 are Looping, Hybrid full and Hybrid context 5. Original-first adds the
source picture on the left and source audio as track 4. GIF excerpts cover seconds 92–98;
stills use second 100. The release also includes the individual context-5 raw and graded videos.
