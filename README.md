# Hybrid Windows for ComfyUI

Experimental H3 hybrid sampling through **two stock KSampler Advanced nodes**.

**Use plain `euler` in both samplers.** The supplied PDD examples use `simple`.
Hybrid sampling is not inherently PDD-only: non-PDD DMD Turbo runs have also
completed with `simple` and `beta57` in the earlier MMH3 hybrid implementation.
See the compatibility table below for the implementation and test scope.

![How the H3 hybrid sampler works: six sequential steps followed by two joint steps](docs/images/h3-hybrid-sampler.jpg)

The diagram illustrates the hybrid concept. Resuming from a finished portion
of a video is not supported in this version.

The examples have three native prompt text boxes, image conditioning,
native PDD LoRA loading, native AV decoding, optional color stabilization and Save Video.

Hybrid sampling supports H3. The decoded-frame color node can be used with other
models and samplers.

## Speed LoRA, sampler and scheduler compatibility

“Completed” means the combination produced output in a recorded test. It does
not establish the best visual quality, long-video stability, or compatibility
with every H3 conditioning mode. The native adapter does not require a PDD
LoRA, but it does require **plain Euler without churn or random inpaint noise**.

| Speed LoRA / setup | Sampler | Scheduler | Steps / split | Evidence and scope |
|---|---|---|---|---|
| [MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors) (PDD) | `euler` | `simple` | 8 / 6+2 | Supplied and tested native Ref2VA workflow. CFG 1; video/audio shifts 12/3. |
| [MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors) (PDD) | `euler` | `simple` | 8 / 6+2 | Supplied and tested native FL2VA workflow. CFG 1; video/audio shifts 12/3. |
| [minimax_h3_dmd_8step_turbo.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/experimental/minimax_h3_dmd_8step_turbo.safetensors) (non-PDD) | `euler` | `simple` | 8 / 6+2 | Completed a 243-frame test with the earlier `MMH3HybridWindowSampler`, Ref2VA base, LoRA strength 1, CFG 1 and shifts 12/3. Not yet verified in this pack's native adapter. |
| [minimax_h3_dmd_8step_turbo.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/experimental/minimax_h3_dmd_8step_turbo.safetensors) (non-PDD) | `euler` | `beta57` | 8 / 6+2 | Completed a 243-frame test with the earlier `MMH3HybridWindowSampler`, using the same base, strength, CFG and shifts. Not yet verified in this pack's native adapter. |
| Other speed LoRAs or scheduler combinations | `euler` | LoRA-specific | LoRA-specific | Not verified. Use the LoRA's intended base model, schedule, step count and CFG; an available dropdown entry is not compatibility evidence. |
| Any LoRA | `euler_ancestral`, Heun, DPM++, UniPC or other non-Euler solvers | Any | Any | Unsupported by this adapter. Its warmup state handling is specific to plain Euler. |

The non-PDD results above were recorded on 2026-09-09. A single-window run
checks the LoRA, scheduler and 6+2 handoff, but does **not** test inter-window
fusion or long-video continuity. The earlier MMH3 sampler supports accepted
prefixes; this standalone native pack does not. Do not transfer a compatibility
claim between those implementations without testing it.

`beta57` comes from **RES4LYF** in the tested installation. It calls ComfyUI's
beta scheduler with alpha **0.5** and beta **0.7**; it is not the stock `beta`
scheduler's default parameterization. Install a provider of that scheduler
before selecting it. Keep the supplied `simple` recipe when reproducing the
native PDD examples.

**Both samplers must use the same scheduler, total step count and model sigma
shifts.** The first sampler's `end_at_step` must equal the second sampler's
`start_at_step`, so the continuation starts at the same noise level where the
first stage stopped. Mixing schedules between stages is unsupported.

For the supplied **8-step PDD LoRA**, keep **8 total steps and CFG 1** in both
nodes, with video/audio sigma shifts **12 / 3**. The default split is **6 + 2**;
splits from **1 + 7 through 7 + 1** are supported. The first sampler adds noise
and returns leftover noise; the second continues that latent with add noise
disabled and finishes denoising.

### Why the last displayed sigma can be nonzero

An eight-step schedule has **nine boundaries**, including its terminal zero.
The first six steps intentionally stop above zero; the final two continue
from that same boundary to zero. For the tested `beta57` schedule with video
shift 12, the values are approximately:

```text
Full:   1.000000 → 0.989805 → 0.968610 → 0.932715 → 0.871939
                 → 0.763505 → 0.563135 → 0.235294 → 0.000000
Warmup: first six transitions, ending at 0.563135
Finish: 0.563135 → 0.235294 → 0.000000
```

Euler evaluates the model at **0.235294** for the last step, then updates the
latent to **sigma 0**. A preview or callback showing 0.235294 therefore does not
mean the output stopped with leftover noise. Inspect the final element of the
complete SIGMAS array, not just the last model-evaluation value.

The earlier `MMH3HybridWindowSampler` explicitly rejects full schedules that
do not end at zero. In this pack's native two-sampler recipe, keep
**return_with_leftover_noise enabled on warmup and disabled on the finish**,
with the finish ending at the total step count. The nonzero warmup endpoint
is required for the handoff; a nonzero terminal endpoint is a different case.

## Included custom nodes

The pack installs **three custom nodes**. The H3 nodes are under
`sampling/hybrid`; Video Color Stabilize is under `image/video`.

| Node | What it does | Used in the examples |
|---|---|---|
| **H3 Hybrid Windows** (`H3HybridWindows`) | Arranges prompt conditioning into overlapping windows. Outputs a `sequential_model` for early sampling with overlap carry, a `joint_model` for finishing with shared overlap predictions, the connected `positive` conditioning, and the calculated `total_frames`. Connect these to the two native KSampler Advanced nodes and the full-timeline latent. | Both Ref2VA and FL2VA |
| **H3 Window ControlNet** (`H3HybridControlNet`) | Applies ComfyUI's native H3 FUN control to the correct source frames for each window. Takes a model, FUN model patch, video VAE and control-frame batch; returns a model with control applied. Provides control strength and start/end timing. It does not extract pose, depth or edges from ordinary footage. | Optional; neither example uses it |
| **Video Color Stabilize** (`VideoColorStabilize`) | Reduces gradual tint and saturation drift in a decoded IMAGE batch, using an opening reference interval and optional aligned source frames. Preserves per-pixel brightness and smooths the correction over time. Returns corrected images. | Both; optional, enabled by default |

The model/LoRA loaders, image loaders, prompt text boxes, H3 conditioning,
KSampler Advanced, VAE decoders and video output nodes in the examples are
provided by ComfyUI itself.

## Optional color stabilization

Connect **VAE Decode → Video Color Stabilize → Create Video → Save Video**.
Audio connects directly to Create Video. Both examples include this wiring and
share one **Output FPS** value between the color node and Create Video.
The color math assumes SDR/sRGB frames with BT.709 luma coefficients.

Defaults are **enabled**, **strength 0.85**, **reference 1–8 seconds** and
**smoothing 2 seconds**. Disable the node or set strength to zero for exact
frame passthrough. The reference interval is preserved; correction ramps in
over two seconds outside it. Use the generated opening as the reference rather
than an appearance-conditioning portrait with potentially different lighting.

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

Restart ComfyUI, then open either example:

- [Ref2VA: reference image](example_workflows/H3%20Hybrid%20-%20native%20KSampler%20Advanced.json).
- [FL2VA: first and last images](example_workflows/H3%20Hybrid%20FL2VA%20-%20native%20KSampler%20Advanced.json).

This pack uses ComfyUI's existing Python dependencies. Model weights are downloaded
separately. Example images are included in [example_workflows/assets](example_workflows/assets/README.md).
Each workflow includes a **READ ME** note with model download links, installation
folders, image setup, sampler settings and instructions for adding windows.

## Examples

Copy the images from `example_workflows/assets/` into `ComfyUI/input/hybrid_windows/`,
or upload them through the native Load Image nodes and select the uploaded files.
Source credits and license links are included in the [assets README](example_workflows/assets/README.md).

Open `example_workflows/H3 Hybrid - native KSampler Advanced.json`. The Ref2VA
example uses `hybrid_windows/ref2va_stock_portrait.jpg`, a stock portrait by
[Nadine Ginzel on Pexels](https://www.pexels.com/photo/portrait-of-a-young-man-in-black-shirt-31428197/).
Its three prompts describe the subject's black shirt, short light brown fringe,
moustache and goatee, with gentle head turns and natural blinking.

The FL2VA example has separate native **Load Image** nodes for the first and
last frames. The first image connects only to window 1; the last image connects
only to window 3. They guide the beginning and end of the **whole video**.
Window 2 continues through overlap carry without an image guide. These
connections stay the same during both sampler stages. Disconnect `last_frame`
from the third conditioning node to generate from a first image alone.

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
samplers and the joint end step. For this native PDD recipe keep total steps 8.
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
directly to Hybrid Windows. FL2VA uses prompts, boundary images and generated
overlap carry.
The adapter expects unmasked AV latents. Preservation of original
source masks is outside this version. Native image guides use frame positions
within their connected window. When adding windows, keep the first-frame
connection on the first window and move the last-frame connection to the new
final window. Set every conditioning node's length to the shared window length.

## How it works

During the first sampler, each window runs its early Euler steps in order.
The final clean prediction supplies temporary video/audio overlap pins to the
next window. Every position keeps the partially denoised state from the first
window that generated it.

That state leaves the first sampler in ComfyUI's ordinary leftover-noise LATENT
format. The second native sampler restores its solver representation with
`add_noise=disable`; it does not substitute the clean prediction or draw fresh
noise. Temporary carry masks stay inside the first call.

For each remaining step, all windows read the same evolving timeline. Their
clean predictions are combined using core's pyramid weights, separately on the
video and audio temporal axes. The native sampler then takes **one global Euler
step**. Overlap can change in this stage. No pixels are blended after decoding.

Sampling uses one native noise draw for the complete timeline, sliced into
windows. Reduced mottling remains an empirical question for real renders,
not a guarantee of this implementation.
