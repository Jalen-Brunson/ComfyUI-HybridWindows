# Hybrid Windows for ComfyUI

Experimental H3 hybrid sampling through **two stock KSampler Advanced nodes**.

![How the H3 hybrid sampler works: six sequential steps followed by two joint steps](docs/images/h3-hybrid-sampler.jpg)

The diagram illustrates the hybrid concept. Resuming from a finished portion
of a video is not supported in this version.

The examples have three native prompt text boxes, image conditioning,
motion control, native PDD LoRA loading, native AV decoding and Save Video.
Only two node types come from this pack:

- **H3 Hybrid Windows** assigns conditioning to windows and supplies separate
  sequential/joint MODEL outputs plus the full timeline's frame count.
- **H3 Window ControlNet** uses core's Fun Control implementation with the
  correct source-frame offset for each window. Its encoded-control cache lasts
  only for one sampling call. Use native **Add Noise to Image** upstream.

This version supports H3. Other video-model families are not implemented yet.

## Installation

From your ComfyUI `custom_nodes` directory:

```bash
git clone https://github.com/Jalen-Brunson/ComfyUI-HybridWindows.git
```

Restart ComfyUI, then open either example:

- [Ref2VA: reference image](example_workflows/H3%20Hybrid%20-%20native%20KSampler%20Advanced.json).
- [FL2VA: first and last images](example_workflows/H3%20Hybrid%20FL2VA%20-%20native%20KSampler%20Advanced.json).

This pack uses ComfyUI's existing Python dependencies. Model weights and test
media are supplied separately through the native loader nodes.

## Examples

Open `example_workflows/H3 Hybrid - native KSampler Advanced.json`. Upload a reference
image and a **24 fps** control video through the native loaders. The example
names `hybrid_man_reference.png` and `hybrid_man_loop_90s.mp4` refer to local
test inputs; media and model weights are not included in this directory.

The FL2VA example has separate native **Load Image** nodes for the first and
last frames. The first image connects only to window 1; the last image connects
only to window 3. They guide the beginning and end of the **whole video**.
Window 2 continues through overlap carry without an image guide. These
connections stay the same during both sampler stages. Disconnect `last_frame`
from the third conditioning node to generate from a first image alone.

The FL2VA test inputs are `hybrid_man_first.png` and `hybrid_man_last.png`;
replace them with your own images and update the prompts to describe them.
First/last conditioning guides the result; it does not paste exact input pixels
into the output.

Default Ref2VA model chain:

1. H3 Ref2VA base: `minimax_h3_ref2va_int8_convrot.safetensors`.
2. Native **LoRA Loader (Model Only)** at 1.0:
   `minimax/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors`.
3. Native H3 video/audio sigma shifts **12 / 3**.
4. Window ControlNet, then Hybrid Windows.

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
calculated total automatically. Increase the control-video trim duration when
making a longer timeline; the example reads its first 30 seconds. A control
source shorter than the requested span repeats its last frame, following core.

The example starts at 832×480, control strength 0.5 and control noise 0.1.
In Ref2VA, all three prompts share the **Load reference image** node. Native H3
conditioning nodes can accept additional reference images. To test without control, connect the sigma
shift MODEL directly to Hybrid Windows; ComfyUI skips the disconnected control
branch. The adapter expects unmasked AV latents. Preservation of original
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
windows. Native float32 LATENT chaining introduces small scaling round-off;
single-window tests compare within 1e-6 absolute / 2e-6 relative tolerance. Reduced mottling remains
an empirical question for real renders, not a guarantee of this implementation.

## Validation

From this directory with ComfyUI's Python environment:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python tests/test_native_chain.py
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 python tests/validate_workflow.py
```

The CPU suite calls actual KSampler Advanced, CFGGuider, ModelPatcher, H3
conditioning/mask/scaling methods, native Euler and the native PDD final layer.
A tiny deterministic model stands in for expensive DiT inference. It checks all
seven handoffs, first/last guide routing and preservation, prompt/control offsets,
overlap pinning/release, prediction fusion, cancellation cleanup and control-cache
lifetime. It does not establish
real-model image quality or CUDA execution.

Workflow validation uses ComfyUI's own prompt validator, checks link types and
both node/group bounds, and verifies that every other node is native. It also
rebuilds both examples against the installed core schemas. A ComfyUI restart is
required after first installing this pack.

A three-window FL2VA GPU smoke test completed with native PDD at 6/2 steps,
first/last images and motion control, exporting 107 frames at 256×256 with audio.
This checks execution and export; long-generation quality still needs visual testing.
