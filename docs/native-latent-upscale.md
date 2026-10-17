# Native Ref2V and FL2V latent upscale

These separate workflows run seven low-resolution native PDD steps, upscale the
clean video estimate, then finish all overlapping windows jointly with DMD.

- [Ref2V workflow](../example_workflows/Ref2V%20Hybrid%20Native%20-%20PDD%20%2B%20DMD%20latent%20upscale.json)
- [FL2V workflow](../example_workflows/FL2V%20Hybrid%20Native%20-%20PDD%20%2B%20DMD%20latent%20upscale.json)

Matching `.api.json` files and a complete
[node dependency manifest](../example_workflows/native_upscale_dependencies.json)
are beside the workflows.

## Install the node dependencies

Use ComfyUI with native MiniMax H3, native PDD LoRA loading, `ManualSigmas`, and
`ComfyMathExpression`. The tested core revision is
`0f74f7fb9f83a78bf46188fd4fd53e6bc44c1ae8`.

Install this HybridWindows revision. It includes `H3HybridWindows`,
`H3HybridCleanLatentUpscale`, `H3HybridLatentUpscale`, `H3HybridWarmupPrior`, and
their sampler handoff helpers.

Also install [Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler).
Its `MinimaxH3LatentUpscaler3D` node is called inside the hybrid upscaler, so it
is required even though it does not appear as a separate workflow node.
From `ComfyUI/custom_nodes`, a reproducible installation is:

```bash
git clone https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler.git
git -C Comfyui_Minimax_h3_latent_Upscaler checkout d7c01b9011f2e8439493f6c02c29995a27df276f
```

Restart ComfyUI after installing. The two workflows use native conditioning and
sampler nodes; they do not require the PDD Acc custom pack, MMH3Tools, KJNodes,
or the additional dependencies of the separate V2V workflow.

## Model files

Paths below are relative to `ComfyUI/models`:

| Folder | File |
| --- | --- |
| `diffusion_models` | `minimax_h3_ref2va_int8_convrot.safetensors` for Ref2V |
| `diffusion_models` | `minimax_h3_fl2va_int8_convrot.safetensors` for FL2V |
| `loras/minimax` | `MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors` for Ref2V |
| `loras/minimax` | `MiniMax-H3-FL2VA-Acc-8Step_comfy.safetensors` for FL2V |
| `loras/minimax` | `minimax_h3_taomate_fl2va_3step_ema_comfyui.safetensors` for both finishing branches |
| `text_encoders` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| `vae` | `minimax_h3_video_vae_int8_convrot.safetensors` |
| `vae` | `minimax_h3_audio_vae_fp32.safetensors` |
| `latent_upscale_models` | `minimax_h3_latent_upscaler_3d_fp32.pth` |

The [upscaler model repository](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
provides the learned upscale weights. Model weights are separate downloads and
are not included in this Git repository.

Copy the bundled images from `example_workflows/assets` into
`ComfyUI/input/hybrid_windows`, or upload your own images in the workflows.
Ref2V uses the same reference in all three windows. FL2V uses the first image
in window 1 and the last image in window 3.

## Sampling and size

| Setting | Value |
| --- | --- |
| Native PDD LoRA | Matching model variant, strength 1 |
| Video / audio sigma shift | 12 / 3 |
| First sampler | Euler, simple eight-step schedule, split after step 7, CFG 1 |
| Upscaler | Clean estimate, 0.8 MP, FP32, temporal chunking off, keep loaded on |
| Second LoRA | Taomate FL2VA DMD, strength 0.8, joint branch only |
| Second sampler | Euler, CFG 1, Disable Noise |
| Finish sigmas | `0.631578922, 0.3158, 0.0000` |
| Window / overlap | 243 / 39 frames; three windows produce 651 frames at 24 fps |

Three sigma points make two finishing updates: the complete recipe is **7 + 2**
steps. DMD replaces the remaining native PDD interval. PDD and DMD are applied
to separate model branches.

The reference recipe is `V2V Upscale PDD+Lightx - faster IO`. It uses the same
FL2VA DMD LoRA on both variants, including Ref2V. This does not establish visual
quality for that combination; the validation below uses a small CPU stand-in.

Change **Final megapixels** to set output size. The defaults produce
1216 × 704 from the Ref2V 832 × 480 warmup, and 704 × 1216 from the FL2V
480 × 832 warmup. FL2V's lower group encodes matching high-resolution guides
and binds them to the same window indices. The joint model still comes from
the original warmup through the upscaler.

Queue the entire graph together. Both the warmup output and denoised output
must reach the upscaler; the final sampler uses its model and latent outputs.
The upscaler retains audio state and reconstructs video noise at the handoff
sigma, so the finishing sampler must not add noise again.

## Validation

From this repository, using ComfyUI's Python environment:

```bash
COMFYUI_PATH=/path/to/ComfyUI python tests/validate_native_upscale.py
```

This validates both API graphs, checks UI link serialization, executes the
seven-step clean upscale handoff with the exact finishing sigmas on a small
CPU stand-in, checks audio preservation and FL2V guide dimensions, and runs
the native-chain and upscale regression tests. Installed model filenames are
checked without loading full H3 weights. Full GPU renders remain untested.
