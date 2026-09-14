# Hybrid Windows for ComfyUI - long Minimax H3 generations without video degradation thorough rolling diffusion and context frame control.

Experimental H3 hybrid sampling through **two stock KSampler Advanced nodes** or a **single Hybrid Window Sampler**.

The bundled hybrid workflows use 5 pinned video context frames inside a
39-frame overlap. Both the native chain and
the single-node sampler expose this setting.

The supplied PDD examples retain **Euler / simple, 8 steps, 6+2**. Hybrid sampling
also accepts other solvers and schedules and does not require PDD or a speed
LoRA. Solver history and stochastic noise need care at the handoff; see the
compatibility table below for the implementation and test scope.

![How the H3 hybrid sampler works: six sequential steps followed by two joint steps](docs/images/h3-hybrid-sampler.jpg)

The diagram illustrates the hybrid concept.

The examples have three native prompt text boxes, image conditioning,
native PDD LoRA loading, native AV decoding, optional color stabilization and Save Video.

Hybrid sampling supports H3. The decoded-frame color node can be used with other
models and samplers.

## V2V Hybrid Sampling with inpainting option

[Download the workflow](example_workflows/V2V%20Hybrid%20Sampling%20with%20inpainting%20option.json) · [Setup and usage guide](docs/v2v-workflow.md) · [API graph](example_workflows/V2V%20Hybrid%20Sampling%20with%20inpainting%20option.api.json)

Transform a source video with reference images and prompts using two native samplers.
Start fresh or continue from accepted chunks: each save includes the complete video
and its matching raw latent master, selected together by project on the next run.
Sapiens2 face/hair segmentation drives the existing blur with mouth protection;
face detection and InsightFace likeness measurement are disabled. Optional inpainting
uses an aligned mask video and selects full overlap pinning automatically.

Defaults are 8 steps, 6 sequential + 2 joint, five context frames, and two new chunks
per queue. The workflow includes editable controls and setup notes. Install the
additional node packs and model files listed in the guide before running it.

### V2V custom-node dependencies

Install all six packs for the supplied V2V workflow, including **ComfyUI-MiniMaxH3Mod**
for its RefMod loader. Install each pack's requirements in ComfyUI's Python environment
and restart. RefMod files are optional; leave the loader slots at `(none)` when unused.

| Node pack | Nodes used by the V2V workflow |
|---|---|
| [ComfyUI-HybridWindows](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows) | `H3HybridWindows`, `H3HybridRunPlan`, `H3HybridResumeLoad`, `H3HybridResumeLatent`, `H3HybridResumeOutput`, `H3HybridSaveRun`, `H3SegmentedVideoBlur` |
| [ComfyUI-MMH3Tools](https://github.com/ckinpdx/ComfyUI-MMH3Tools) | `MMH3ReferenceMultiPrompt`, `MMH3StreamingEncode`, `MMH3PackAV`, `H3RefModCondSetApply`; optional `MMH3ImageList` for multiple pictures; `MMH3JoinAV` is called by the continuation helper |
| [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | `VHS_LoadVideoFFmpegPath` for aligned source, mask, control and accepted-prefix loading |
| [ComfyUI-Sapiens2](https://github.com/kijai/ComfyUI-Sapiens2) | `Sapiens2Loader`, `Sapiens2Seg`, `Sapiens2SegExtract` |
| [h3_face_tools](https://github.com/Jalen-Brunson/h3_face_tools) | `FaceAnonymizeVideo`, called by the segmentation-blur helper with face detection and likeness measurement disabled |
| [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) | `MiniMaxH3RefModsLoader` loads saved RefMods; MMH3Tools' `H3RefModCondSetApply` applies them to every window |

The remaining workflow nodes are supplied by ComfyUI. Video metadata uses ComfyUI's
existing PyAV dependency. VideoHelperSuite's `imageio-ffmpeg` dependency supplies
its video/audio decoder in the standard pip installation on Windows, macOS and
Linux. **No separate FFmpeg/ffprobe installation or PATH configuration is needed
for this workflow on those installs.** Model downloads and installation folders
are in the [V2V setup guide](docs/v2v-workflow.md#install-nodes-and-models).

### Use a RefMod without a reference image

Mute or bypass **Reference image — Picture 1**, select a RefMod in its loader, and
write the prompt using the subject description shown in **RefMod prompt hint**.
Remove `<Picture 1>` when no image is connected; keep `<Video 1>` for the source
motion control. For example: `A person with short dark hair follows the movement
and timing in <Video 1>.` Put saved RefMod files in `models/refmods/` and refresh
the loader. Image references, RefMods, or both are supported.

## Example: Looping Sampler vs Hybrid on a two-minute clip

The comparison now includes **Hybrid with 5 pinned video context frames**, alongside the
original **full-overlap Hybrid** and **MMH3 Looping Sampler** recordings. Left to right:
the motion-reference clip (`stream_00170`), Looping, Hybrid (full), and Hybrid context 5.
The same reference picture replaces the source performer in all three renders.

All runs use 2895 frames / 120.625 s at 768x576 and 24 fps, 14 windows of 243 frames with a
39-frame overlap, seed 123, Ref2VA int8, the native PDD 8-step LoRA at 1.0, Euler / simple,
CFG 1, plain SDPA attention, and one prompt for every window. Looping uses 8 sequential
steps; both hybrids use 6 sequential + 2 joint steps. Context 5 changes only the pinned
video carry during warmup; the overlap stays at 39 frames.

[![Original | Looping | Hybrid full | Hybrid context 5 at 1:32](docs/examples/original_looping_hybrid_92s.gif)](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/releases/download/examples-stream00170/stream00170_Original_Looping_Hybrid.mp4)

*Six-second excerpt at 1:32, in window 12 of 14. Click for the full video. Audio tracks
1–3 are Looping, Hybrid (full), and Hybrid context 5; track 4 is the original audio.*

[![Looping | Hybrid full | Hybrid context 5 with per-window metrics](docs/examples/looping_vs_hybrid_metrics_92s.gif)](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/releases/download/examples-stream00170/stream00170_Looping_vs_Hybrid_chunk_metrics.mp4)

*The same excerpt with the existing chunk-quality panels. Window 1 is each render’s
anchor; green = OK, amber = WATCH. Click for the full video, with one audio track per arm.*

Full-length comparisons, with the new context-5 arm added:
[clean](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/releases/download/examples-stream00170/stream00170_Looping_vs_Hybrid_clean.mp4) ·
[with metrics](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/releases/download/examples-stream00170/stream00170_Looping_vs_Hybrid_chunk_metrics.mp4) ·
[Original first](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/releases/download/examples-stream00170/stream00170_Original_Looping_Hybrid.mp4) ·
colour-corrected: [clean](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/releases/download/examples-stream00170/stream00170_Looping_vs_Hybrid_color_clean.mp4) ·
[with metrics](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/releases/download/examples-stream00170/stream00170_Looping_vs_Hybrid_color_chunk_metrics.mp4) ·
[Original first](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/releases/download/examples-stream00170/stream00170_Original_Looping_Hybrid_color.mp4).

Individual context-5 render: [raw](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/releases/download/examples-stream00170/hybrid_native_context5_6p2_00001_.mp4) ·
[colour-corrected](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/releases/download/examples-stream00170/hybrid_native_context5_6p2_00001__color.mp4).

Stills at 1:40: [Original + all three renders](docs/examples/original_looping_hybrid_100s.png) ·
[metrics](docs/examples/looping_vs_hybrid_metrics_100s.png) ·
[colour-corrected](docs/examples/original_looping_hybrid_color_100s.png) ·
[colour-corrected metrics](docs/examples/looping_vs_hybrid_color_metrics_100s.png).

| Measured on the finished videos | Looping (8) | Hybrid full (6+2) | Hybrid context 5 (6+2) |
|---|---|---|---|
| GPU | H200 | H200 | H100 80GB |
| Pure sampling time | 43:17 | 43:14 | 45:00 |
| ArcFace likeness, whole clip (median / p10) | 0.649 / 0.581 | 0.736 / 0.693 | 0.743 / 0.714 |
| Likeness, window 1 → window 14 | 0.72 → 0.56 | 0.75 → 0.71 | 0.76 → 0.73 |
| Fine texture at window 14 | ×1.56 | ×1.45 | ×1.29 |
| Skin colour patchiness (mottle) at window 14 | ×1.54 | ×1.00 | ×1.08 |
| Contrast at window 14 | ×1.18 | ×1.16 | ×1.09 |
| Saturation at window 14 | ×1.17 | ×1.13 | ×1.02 |
| First WATCH verdict | window 7 | window 9 | none |

Context 5 stayed **OK in all 14 windows** in this readout, with less texture, contrast and
saturation drift than the archived full-overlap hybrid. Likeness was slightly higher.
Quality ratios use each clip’s own first-window anchor and the source trajectory, so they
are not absolute quality scores. This is one seed on one clip; the new run used the current
implementation and an H100, while the September 10 baselines used an H200. All arms use
112 model evaluations, but the timings are not a direct hardware-matched speed comparison.

The colour-corrected variants apply the same chroma-only settings to all three renders:
tint and saturation follow the opening 1–8 seconds and the source’s colour trajectory;
brightness, contrast, texture and audio are left unchanged. The runnable comparison
hybrid graph now selects context 5. [Exact settings, measurements and original baseline
provenance](docs/examples/compare_stream00170/).

## Single-sampler reference-image example

Open [H3 Hybrid R2V - single sampler.json](example_workflows/H3%20Hybrid%20R2V%20-%20single%20sampler.json) in ComfyUI. An [API graph](example_workflows/H3%20Hybrid%20R2V%20-%20single%20sampler.api.json) is included.

This sampler moved from **ComfyUI-MMH3Tools** into this repository. Its node ID remains `MMH3HybridWindowSampler`, so existing workflows keep their connections. Update both packs and restart to avoid duplicate registration. The single sampler requires MMH3Tools for its shared window, conditioning and mask helpers; the original two-KSampler examples do not.

The example has **16 native nodes**, **MMH3 Cond To Set**, and **H3 Hybrid Window Sampler**. Native H3 Ref2VA conditioning receives the loaded reference image; Cond To Set reuses that conditioning across three windows. Native noise, Euler selection and sigma scheduling feed the sampler, followed by native video/audio decoding and Save Video.

1. Copy the bundled [reference portrait](example_workflows/assets/ref2va_stock_portrait.jpg) into `ComfyUI/input/hybrid_windows/`, or upload your own image in Load Image. Asset credits are in the assets folder.
2. Select the Ref2VA base, matching PDD LoRA, H3 text encoder and both VAEs listed below. Edit the native conditioning prompt; `<Picture 1>` identifies the reference.
3. Queue to generate **651 frames at 832 × 480, 24 fps** with **243-frame windows / 39-frame overlap**, **6 sequential + 2 joint Euler steps**, CFG 1, and video/audio shifts 12/3. Output saves under `output/video/HybridSampler_Ref2VA`.

Set `sequential_steps = 8` for the sequential baseline. For another duration, match Cond To Set's count and the empty latent length: `total = window + (count - 1) × (window - overlap)`. Keep conditioning and latent dimensions identical. Connect the model before any Context Windows node; this sampler owns windowing. The sample starts fresh with accepted prefix and start window both zero.

Rebuild with `python build_sampler_workflow.py`. Validate with `python tests/validate_sampler_workflow.py` and run the sampler state tests with `python tests/test_hybrid_sampler.py` from this repository in the ComfyUI environment. Graph validation and CPU sampler tests do not establish rendered visual quality.

## Single hybrid sampler

**H3 Hybrid Window Sampler** (`MMH3HybridWindowSampler`) and its joint-stage helpers are provided by this repository. It uses the public MMH3Tools window/AV utilities and conditioning type; it does not require `MMH3JointWindowSampler` or local-only MMH3Tools files. Keep the node ID when updating existing workflows. The default is six sequential steps followed by two joint steps. Select `overlap_pin = tail_custom` and `context_frames = 5` for the same pinned-context behavior as the native flow. The published reference-image example selects these values; existing workflows retain `full` pinning until changed.

## Speed LoRA, sampler and scheduler compatibility

“Completed” means the combination produced output in a recorded test. It does
not establish the best visual quality, long-video stability, or compatibility
with every H3 conditioning mode. Neither hybrid sampler requires PDD, a speed
LoRA, or Euler. The single-node sampler still uses CFG 1 and a finite, strictly
descending sigma schedule ending at zero. Random inpaint noise is unsupported
because it replaces the fixed noise used to restore pinned rows.

| Speed LoRA / setup | Sampler | Scheduler | Steps / split | Evidence and scope |
|---|---|---|---|---|
| [MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors) (PDD) | `euler` | `simple` | 8 / 6+2 | Supplied and tested native Ref2VA workflow. CFG 1; video/audio shifts 12/3. |
| [MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors) (PDD) | `euler` | `simple` | 8 / 6+2 | Supplied and tested native FL2VA workflow. CFG 1; video/audio shifts 12/3. |
| [minimax_h3_dmd_8step_turbo.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/experimental/minimax_h3_dmd_8step_turbo.safetensors) (non-PDD) | `euler` | `simple` | 8 / 6+2 | Completed a 243-frame test with the earlier `MMH3HybridWindowSampler`, Ref2VA base, LoRA strength 1, CFG 1 and shifts 12/3. Not yet verified in this pack's native adapter. |
| [minimax_h3_dmd_8step_turbo.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/experimental/minimax_h3_dmd_8step_turbo.safetensors) (non-PDD) | `euler` | `beta57` | 8 / 6+2 | Completed a 243-frame test with the earlier `MMH3HybridWindowSampler`, using the same base, strength, CFG and shifts. Not yet verified in this pack's native adapter. |
| Other speed LoRAs or scheduler combinations | Model/LoRA-specific | Model/LoRA-specific | Model/LoRA-specific | Accepted, quality not verified. Use the intended base, schedule, step count and CFG. |
| Base model without acceleration | `euler` | `beta57` | 10 / 6+4 | Completed a six-chunk native run; this configuration still developed mottling. More steps did not establish a quality fix. |
| Any compatible model / LoRA | `heun`, `dpm_2` | Matching schedule | Any valid split | CPU tests verify exact continuation of the solver state in isolation; full H3 visual quality is not established. |
| Any compatible model / LoRA | Multistep or stochastic solvers | Matching schedule | Any valid split | Accepted. Multistep history restarts at the handoff; stochastic draws are not preserved as one uninterrupted noise process. |

**Both samplers must use the same scheduler, total step count and model sigma
shifts.** The first sampler's end step must equal the second sampler's start
step. Mixing schedules between stages is unsupported.

Set **Switch at step = 8** for the **8 + 0 sequential baseline**. Each window
finishes before passing its overlap to the next; the second stock sampler
passes the completed latent through unchanged. The shared **Total steps** value
also connects to Hybrid Windows. Use this baseline to compare sequential carry
with joint finishing while keeping the seed, prompts and inputs fixed.

## Included custom nodes

The pack installs ten custom nodes. Sampling nodes are under `sampling/hybrid`,
continuation helpers under `sampling/hybrid/continuation`, and color/blur nodes
under `image/video`.

| Node | What it does | Used in the examples |
|---|---|---|
| **H3 Hybrid Window Sampler** (`MMH3HybridWindowSampler`) | Single-node sequential warmup and joint finish, with adjustable pinned context, source masks and accepted-prefix preservation. Requires MMH3Tools. | Single-sampler Ref2VA |
| **H3 Hybrid Windows** (`H3HybridWindows`) | Arranges prompt conditioning into overlapping windows. Outputs a `sequential_model` for early sampling with overlap carry, a `joint_model` for finishing with shared overlap predictions, the connected `positive` conditioning, the calculated `total_frames`, the `latent` to sample, and a `report`. Connect the two models to the two native sampler nodes. Optional inputs cover a production graph: an MMH3 `cond_set` instead of the prompt sockets, a source `latent` with `denoise_mask` / `audio_denoise_mask` for v2v inpainting, `accepted_prefix_frames` + `start_window` for an accepted-prefix resume, and `noise_mode`. | Both Ref2VA and FL2VA |
| **H3 Window ControlNet** (`H3HybridControlNet`) | Applies ComfyUI's native H3 FUN control to the correct source frames for each window. Takes a model, FUN model patch, video VAE and control-frame batch; returns a model with control applied. Provides control strength and start/end timing. It does not extract pose, depth or edges from ordinary footage. | Optional; neither example uses it |
| **Video Color Stabilize** (`VideoColorStabilize`) | Reduces gradual tint and saturation drift in a decoded IMAGE batch, using an opening reference interval and optional aligned source frames. Preserves per-pixel brightness and smooths the correction over time. Returns corrected images. | Both; optional, enabled by default |

The V2V workflow also includes **V2V Run Plan**, **Load V2V Continuation**,
**Prepare V2V and Inpainting**, **Assemble V2V Continuation**, **Save V2V Video
and Continuation**, and **Blur Segmented Face and Hair**. See the
[V2V guide](docs/v2v-workflow.md) for their controls and dependencies.

The Ref2VA/FL2VA image-conditioned examples use ComfyUI's native model/LoRA loaders,
image loaders, prompt text boxes, H3 conditioning, KSampler Advanced, VAE decoders
and video output nodes. The V2V workflow also uses the custom-node dependencies
listed above.

## Source masks, resume and per-window noise

With none of these connected the node behaves exactly as the examples show. They
exist so a production graph -- v2v inpainting with an accepted-prefix resume --
can run on this chain instead of the single-node sampler.

- **`cond_set`** takes one conditioning per window from MMH3 Reference
  (Multi-Prompt), instead of the `positive_N` sockets. Connect one or the other.
- **`latent`** is the master to sample: a source encode for v2v, or a
  continuation master on resume. The node composes the masks and the accepted
  prefix into one per-row noise mask, so **sample the node's `latent` output**,
  not the master you connected.
- **`denoise_mask` / `audio_denoise_mask`** (white regenerates, black keeps the
  source) are reduced onto the latent grid by `denoise_mask_mode`. Pinned rows
  are held through both stages: the joint stage keeps the model looking at the
  clean source, so they are never re-injected from a half-denoised latent.
- **`accepted_prefix_frames`** holds a finished head fixed. Windows that lie
  entirely inside it skip the warm-up, and the output takes those rows from the
  input. It must be 0 or 5+17k frames.
- **`start_window`** is this run's first global window index (the resume shift);
  it seeds the per-window noise.
- **`noise_mode`**: `global` slices one draw, as before. `per_window` draws
  `seed + start_window + window index` per window, the same draws the
  single-node sampler makes, so a graph can be moved between the two samplers
  and compared.

Both samplers must run in one queue. The joint stage resumes the exact state the
warm-up left, and a restart in between loses it -- it says so rather than
sampling something wrong. Take the warm-up's `output`, never `denoised_output`.

A short latent is accepted: when the master is shorter than
`window + (count-1) * (window - overlap)` the last window slides back to end at
the clip, the way ComfyUI's own static window schedule does. That is what a
continuation master looks like on a project's last part.

## Optional color stabilization

Connect **VAE Decode → Video Color Stabilize → Create Video → Save Video**.
Audio connects directly to Create Video. Both examples include this wiring and
share one **Output FPS** value between the color node and Create Video.
The color math assumes SDR/sRGB frames with BT.709 luma coefficients.

Defaults are **enabled**, **strength 0.85**, **reference 1–8 seconds** and
**smoothing 2 seconds**, with **max_tint_shift 0.06**. Disable the node or set strength to zero for exact
frame passthrough. The reference interval is preserved; correction ramps in
over two seconds outside it. Use the generated opening as the reference rather
than an appearance-conditioning portrait with potentially different lighting.

For strong accumulated tint drift, use strength **1.0**. `max_tint_shift`
limits each Cb/Cr correction before strength is applied; 0.06 allows about
15 levels on a 0–255 scale. The previous fixed 0.025 limit could leave a
visible cast on long clips. This adjusts decoded output, not the latent
carried into the next generation window.

Without `source_frames`, the target is the opening's median tint and color
spread. Optionally connect original video frames **before noise or effects**,
with the same frame count, FPS and starting time as the generated images.
Resolution may differ. This follows source tint changes relative to its own
opening; source variation may raise the saturation target but does not lower
it below the generated opening. The node rejects mismatched frame counts to
avoid applying corrections at the wrong time. Leave this input disconnected
in the supplied image-conditioned examples.

This is intended for a **continuous shot**, processed as one complete IMAGE
batch. It keeps the same reference across generation-window boundaries.
Split footage at scene cuts. Whole-frame statistics also respond to changes
in composition, and without source frames the node can reduce intentional
color changes; lower strength or bypass it when needed.

Only chroma is adjusted: brightness, spatial detail and audio are not graded.
At saturated pixels, the color change is limited to stay within RGB gamut
while preserving luma. Sampling and latent carry are unchanged. This treats
visible color drift, not its inference cause or spatial mottling.

Processing uses CPU work on one frame at a time, with small temporal statistics
and one output batch. The complete input and output IMAGE batches still need
RAM; this node does not turn decoding/export into a streaming workflow. Color
is corrected before video encoding, avoiding a separate recompression pass.

## Installation

From your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/Jalen-Brunson/ComfyUI-HybridWindows.git
```

Restart ComfyUI, then open an example:

- [V2V Hybrid Sampling with inpainting option](example_workflows/V2V%20Hybrid%20Sampling%20with%20inpainting%20option.json); install the [V2V dependencies](#v2v-custom-node-dependencies) above.
- [R2V (Ref2VA): reference image](example_workflows/H3%20Hybrid%20R2V%20-%20native%20KSampler%20Advanced.json).
- [FL2VA: starting image and prompts](example_workflows/H3%20Hybrid%20FL2VA%20-%20native%20KSampler%20Advanced.json).
- [FL2VA: optional recurring ending guide](example_workflows/H3%20Hybrid%20FL2VA%20-%20recurring%20end%20guide.json).

This pack uses ComfyUI's existing Python dependencies. Model weights are downloaded
separately. Example images are included in [example_workflows/assets](example_workflows/assets/README.md).
Each workflow includes a **READ ME** note with model download links, installation
folders, image setup, sampler settings and instructions for adding windows.

## Examples

Copy the images from `example_workflows/assets/` into `ComfyUI/input/hybrid_windows/`,
or upload them through the native Load Image nodes and select the uploaded files.
Source credits and license links are included in the [assets README](example_workflows/assets/README.md).

R2V means reference-to-video; this example uses the H3 Ref2VA model.

Open `example_workflows/H3 Hybrid R2V - native KSampler Advanced.json`. The Ref2VA
example uses `hybrid_windows/ref2va_stock_portrait.jpg`, a stock portrait by
[Nadine Ginzel on Pexels](https://www.pexels.com/photo/portrait-of-a-young-man-in-black-shirt-31428197/).
Its three prompts describe the subject's black shirt, short light brown fringe,
moustache and goatee, with gentle head turns and natural blinking.

The three prompts describe an opening, continuation and ending. Keep camera
distance, framing, appearance and lighting consistent, and describe only the
current portion of the action in each box. Repeating a complete camera move
in every window can ask the model to restart that move. The example prompts
favor restrained motion and a continuous composition.

The main FL2VA example uses a **starting image and three prompts**. The first
image connects only to window 1; later windows continue through generated
latent overlap. An optional ending-image loader is included but disconnected.
No control video is required.

The **recurring end guide** variant connects the last image to `last_frame` on
all three native encoders. Window 1 moves into that composition; windows 2 and
3 continue the established shot with the same guide. It reaches the ending
view early, so use it when a stable target composition matters more than
arriving at that view only at the final moment.

Connecting the last image only to the final window is possible, but caused
late reframing in our 27-second tests. It is not the recommended setup for a
continuous long shot. The plain native first/last configuration worked in a
single-window test. Neither configuration guarantees exact endpoint pixels.
No generated frames are turned into keyframe guides; overlap continuity uses
latent masks.

The bundled FL2VA inputs are `hybrid_windows/fl2va_stock_first.png` and
`hybrid_windows/fl2va_stock_last.png`, two frames from a
[Kampus Production stock clip](https://www.pexels.com/video/man-wearing-black-long-sleeve-polo-8189169/).
They show the same man at a white desk, with a gradual change in camera angle.
Replace them with your own images and update the prompts to describe them.
First/last conditioning guides the result; it does not paste exact input pixels
into the output.

Default Ref2VA model chain:

1. H3 Ref2VA base: `minimax_h3_ref2va_int8_convrot.safetensors`.
2. Native **LoRA Loader (Model Only)** at 1.0:
   [minimax/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors).
3. Native H3 video/audio sigma shifts **12 / 3**.
4. Hybrid Windows.

Use an unbaked base with this LoRA to avoid applying the PDD changes twice.
The FL2VA flow substitutes `minimax_h3_fl2va_int8_convrot.safetensors` and
[minimax/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors). It uses native
**MiniMax H3 Image to Video** conditioning with the same sampler settings.

| Setting | Sequential KSampler Advanced | Joint KSampler Advanced |
|---|---|---|
| Model | `sequential_model` | `joint_model` |
| Latent | Native empty full-timeline AV latent | First sampler's output |
| Add noise | Enable | Disable |
| Steps | 8 | 8 |
| Start / end | 0 / 6 | 6 / 8 |
| Return with leftover noise | Enable | Disable |
| Sampler / scheduler / CFG | euler / simple / 1 | euler / simple / 1 |

`Switch at step` drives both sides of the handoff. `Total steps` drives both
samplers, Hybrid Windows and the joint end step. For this native PDD recipe
keep total steps 8.
Each sampler's step numbers refer to the **same complete schedule**.

The three prompts correspond to three overlapping windows. At the default
243 frames / 39 overlap, their frame spans are **0–242, 204–446, 408–650**.
Output length is **651 frames / 27.125 seconds at 24 fps**. Window and overlap
values snap up to H3's 17k+5 frame grid. The native empty latent receives the
calculated total automatically.

Ref2VA starts at 832×480; FL2VA starts at 480×832 to match its portrait images.
In Ref2VA, all three prompts share the
**Load reference image** node. Native H3 conditioning nodes can accept
additional reference images. Both examples connect the sigma-shift MODEL
directly to Hybrid Windows. FL2VA uses prompts, a starting image and generated
overlap carry, with an optional recurring ending guide.
These examples use unmasked AV latents. For source masks or accepted-prefix
resume, connect the source latent to Hybrid Windows and sample its prepared
`latent` output. Partial context currently requires full-frame regeneration
outside an accepted prefix; use `overlap_pin = full` when protecting fresh source
video with an inpaint mask. Native image guides use frame positions within their
connected window. When adding windows, keep the first-frame
connection only on the first window. In the recurring-end variant, also
connect the ending guide to each added encoder. Set every conditioning node's
length to the shared window length.

## How it works

During the first sampler, each window runs its early sampling steps in order.
The final clean prediction supplies temporary video/audio overlap pins to the
next window. The image-conditioned examples pin only the last five video frames
of the overlap; all overlap frames remain visible. Audio overlap remains pinned.
Every position keeps the partially denoised state from the first window that
generated it.

That state leaves the first sampler in ComfyUI's ordinary leftover-noise LATENT
format. The second native sampler restores its solver representation with
`add_noise=disable`; it does not substitute the clean prediction or draw fresh
noise. Temporary carry masks stay inside the first call.

For each remaining step, all windows read the same evolving timeline. Their
clean predictions are combined using core's pyramid weights, separately on the
video and audio temporal axes. The selected solver updates the shared timeline
from these predictions. Overlap can change in this stage. No pixels are blended after decoding.

Sampling uses one native noise draw for the complete timeline, sliced into
windows. Reduced mottling remains an empirical question for real renders,
not a guarantee of this implementation.

### Previews and progress during the first sampler

Each window's steps are reported to the sampler node as one continuous run:
`windows x steps` total, counted from the first sampled window. The preview
latent is the whole timeline with the current window's prediction written into
it, so a preview shows finished windows, the window being sampled, and the
untouched remainder. Windows inside an accepted prefix are skipped and report
nothing. ComfyUI's built-in previewers render the first latent frame of what
they are given, so on a resume that frame belongs to the accepted prefix and
does not change; a preview node that decodes a clip, such as KJNodes' **Model
Preview Override**, shows the window.

The window loop is registered as the outermost `OUTER_SAMPLE` wrapper of both
stage models. Other wrappers -- preview overrides, caches -- therefore run once
per sampled window and receive that window's noise, sigmas and latent shapes,
whether they were added before or after this node in the graph.

## Research credit

Credit to **David Ruhe, Jonathan Heek, Tim Salimans, and Emiel Hoogeboom** for
[Rolling Diffusion Models](https://proceedings.mlr.press/v235/ruhe24a.html)
(ICML 2024), an inspiration for temporal generation through sliding-window
denoising. This H3 adaptation uses sequential Euler warmup followed by a joint
window finish.
