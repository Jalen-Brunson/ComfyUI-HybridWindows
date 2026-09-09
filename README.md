# Hybrid Windows for ComfyUI

Experimental H3 hybrid sampling through **two stock KSampler Advanced nodes**.

**Use plain `euler` with `simple` in both samplers.** Other samplers are
unsupported; other schedulers have not been verified for this H3/PDD setup.

![How the H3 hybrid sampler works: six sequential steps followed by two joint steps](docs/images/h3-hybrid-sampler.jpg)

The diagram illustrates the hybrid concept. Resuming from a finished portion
of a video is not supported in this version.

The examples have three native prompt text boxes, image conditioning,
native PDD LoRA loading, native AV decoding and Save Video.

This version supports H3. Other video-model families are not implemented yet.

## Supported samplers and schedulers

| Setting | Support |
|---|---|
| **Sampler: `euler`** | The only supported sampler in both stages. Use plain Euler without churn or additional random inpaint noise. |
| **Other samplers** | Unsupported and rejected by the adapter, including `euler_ancestral`, Heun, DPM++ and UniPC. |
| **Scheduler: `simple`** | The verified scheduler for the supplied H3/PDD workflows. Select it in both KSampler Advanced nodes. |
| **Other schedulers** | Not verified for this setup. The adapter does not block them, but their presence in ComfyUI's dropdown does not establish compatibility. |

**Both samplers must use the same scheduler, total step count and model sigma
shifts.** The first sampler's `end_at_step` must equal the second sampler's
`start_at_step`, so the continuation starts at the same noise level where the
first stage stopped. Mixing schedules between stages is unsupported.

For the supplied **8-step PDD LoRA**, keep **8 total steps and CFG 1** in both
nodes, with video/audio sigma shifts **12 / 3**. The default split is **6 + 2**;
splits from **1 + 7 through 7 + 1** are supported. The first sampler adds noise
and returns leftover noise; the second continues that latent with add noise
disabled and finishes denoising.

## Included custom nodes

The pack installs **two custom nodes**, both under `sampling/hybrid`:

| Node | What it does | Used in the examples |
|---|---|---|
| **H3 Hybrid Windows** (`H3HybridWindows`) | Arranges prompt conditioning into overlapping windows. Outputs a `sequential_model` for early sampling with overlap carry, a `joint_model` for finishing with shared overlap predictions, the connected `positive` conditioning, and the calculated `total_frames`. Connect these to the two native KSampler Advanced nodes and the full-timeline latent. | Both Ref2VA and FL2VA |
| **H3 Window ControlNet** (`H3HybridControlNet`) | Applies ComfyUI's native H3 FUN control to the correct source frames for each window. Takes a model, FUN model patch, video VAE and control-frame batch; returns a model with control applied. Provides control strength and start/end timing. It does not extract pose, depth or edges from ordinary footage. | Optional; neither example uses it |

The model/LoRA loaders, image loaders, prompt text boxes, H3 conditioning,
KSampler Advanced, VAE decoders and video output nodes in the examples are
provided by ComfyUI itself.

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
   `minimax/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors`.
3. Native H3 video/audio sigma shifts **12 / 3**.
4. Hybrid Windows.

Use an unbaked base with this LoRA to avoid applying the PDD changes twice.
The FL2VA flow substitutes `minimax_h3_fl2va_int8_convrot.safetensors` and
`minimax/MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors`. It uses native
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
