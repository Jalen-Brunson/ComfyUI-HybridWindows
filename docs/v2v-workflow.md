# V2V Hybrid Sampling with inpainting option

Transform a source video using reference pictures and prompts, generate it in manageable parts, and continue from accepted chunks. Face and hair blur uses **Sapiens2 segmentation**, with the mouth protected. The workflow uses two native KSampler Advanced nodes.

[Download the workflow](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/blob/main/example_workflows/V2V%20Hybrid%20Sampling%20with%20inpainting%20option.json) · [API graph](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/blob/main/example_workflows/V2V%20Hybrid%20Sampling%20with%20inpainting%20option.api.json)

## Start a video

1. Set **Source video path** to your footage, including an audio track. Paths can be absolute or relative to ComfyUI, such as `input/source.mp4`.
2. Give it a **Project name**. Use a different name for another source or alternative settings.
3. Choose an image, a saved RefMod, or both for the desired appearance, then write your **Prompt**. For an image, use **Reference image — Picture 1** and refer to it as `<Picture 1>`. For RefMod only, mute or bypass that image loader and describe the subject using **RefMod prompt hint**; remove `<Picture 1>` from the prompt. `<Video 1>` remains the source-derived motion control.
4. In **Run controls**, leave **accepted_chunks = 0**, choose **new_chunks**, and set your output width/height. Defaults generate two chunks: 447 frames / 18.625 seconds at 832×480, 24 fps, if the source is long enough.
5. Leave **Enable inpainting = false** for full-frame regeneration. **Enable segmentation blur = true** uses the face/hair mask; disable it to skip segmentation completely.
6. Select the model files described below and **Queue**.

The output appears under `ComfyUI/output/video/V2V_Hybrid…mp4`. **Saved run and next continuation setting** tells you how many chunks to accept next. Each successful save also writes its matching full latent master under `output/latents/hybrid_windows/`.

The defaults are **8 sampling steps, 6 sequential + 2 joint, Euler/simple, CFG 1**, with **243-frame windows, 39-frame overlap and 5 context frames**. Changing **Switch to joint sampling at step** to 8 gives an 8/0 sequential comparison. Keep both stages on the same schedule; the linked step and seed controls do this automatically.

<!-- workflow-panel -->
# Install nodes and models

Use current ComfyUI with native MiniMax H3 and continuous per-row mask support. Install these public node packs with ComfyUI Manager or clone their repositories into `ComfyUI/custom_nodes/`, install their requirements in ComfyUI's Python environment, then restart:

- [ComfyUI-HybridWindows](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows): sampling and the continuation/segmentation-blur workflow helpers.
- [ComfyUI-MMH3Tools](https://github.com/ckinpdx/ComfyUI-MMH3Tools): reference conditioning, streaming source encode and AV utilities.
- [ComfyUI-VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite): aligned video/audio loading.
- [ComfyUI-Sapiens2](https://github.com/kijai/ComfyUI-Sapiens2): face, hair and mouth segmentation.
- [h3_face_tools](https://github.com/Jalen-Brunson/h3_face_tools): the segmentation-only blur implementation. This workflow disables its face detector and likeness measurement; no InsightFace model is loaded. The pack's requirements may still install InsightFace for its other nodes.
- [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod): `MiniMaxH3RefModsLoader` loads saved RefMods, which MMH3Tools' `H3RefModCondSetApply` applies to every window. Install the pack for the supplied graph; using RefMod files is optional. Leave its slots at `(none)` to skip them.

**No separate FFmpeg/ffprobe setup is needed with the standard pip dependencies on Windows, macOS or Linux.** The run planner reads video metadata through ComfyUI's existing PyAV dependency. VideoHelperSuite uses the FFmpeg binary provided by `imageio-ffmpeg` for video/audio loading; it does not need that binary on PATH. Install the node-pack requirements listed above, then restart ComfyUI.

Download the model files below and select them in the corresponding loaders. These paths are relative to the ComfyUI folder.

| Model | Download | Folder |
|---|---|---|
| H3 Ref2VA | [minimax_h3_ref2va_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors) | `models/diffusion_models/` |
| Native eight-step PDD LoRA | [MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/loras/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors) | `models/loras/` |
| H3 text encoder | [qwen3vl_32b_minimax_h3_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) | `models/text_encoders/` |
| Video VAE | [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/minimax_h3_video_vae_int8_convrot.safetensors) | `models/vae/` |
| Audio VAE | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `models/vae/` |
| Segmentation | [sapiens2_5b_seg.safetensors](https://huggingface.co/facebook/sapiens2-seg-5b/blob/main/sapiens2_5b_seg.safetensors) | `models/sapiens2/` |

Use CLIP type **minimax**, LoRA strength **1**, and video/audio sigma shifts **12/3**. The segmentation checkpoint is only executed when blur is enabled. A compatible smaller Sapiens2 segmentation model can be selected instead; pose and normals checkpoints are unsuitable.

When using a reference image, upload your own picture, or copy the bundled `example_workflows/assets/ref2va_stock_portrait.jpg` into `ComfyUI/input/hybrid_windows/`. See the assets folder for its credit. The shared workflow contains placeholder source/mask/control paths; it does not contain private footage or machine-specific paths.

<!-- workflow-panel -->
# Continue or redo a section

**Accepted chunks** means the number of completed chunks to keep. It is not the chunk to generate next, a frame count, or a sampling-step count.

For example, generate two chunks with **accepted_chunks = 0 / new_chunks = 2**. After reviewing the result, set **accepted_chunks = 2 / new_chunks = 2** and queue again. The new save contains the original two accepted chunks plus two new ones. Then use **accepted_chunks = 4** for another continuation.

**Select continuation** defaults to **Latest completed run**, scoped to your Project name. It selects the saved video and the exact raw latent master that produced it. To branch from an older result, refresh ComfyUI and choose that run in its dropdown, labeled with the project, chunk count and video filename. A smaller accepted count regenerates the suffix after that point.

Keep the same source file, source start, dimensions, window size and overlap across a chain. Changing prompts for later chunks is supported. Use one prompt to repeat it throughout, or a `|` separated list indexed from global chunk 1; the run planner automatically selects the correct portion, including the pinned context window. Keep the same seed when you want stable global per-window seeds.

The source, mask and optional second control loaders automatically advance together. Only the current run's span is loaded and encoded. A resumed run includes one already accepted window for conditioning; that window is pinned in sampling and removed exactly once before appending the new frames. The accepted decoded prefix and its waveform are preserved before normal MP4 encoding.

Every save contains a full video and full raw latent master beginning at the project start. Keep **both** the video and its master/index files if you want to continue later. Incomplete saves are not offered as completed runs. These continuation records start with runs saved by this workflow; older local diagnostic masters are not automatically imported.

**new_chunks = 0** generates everything remaining. **max_frames = 0** uses the remaining source; a positive value caps the full project. The planner rounds the final project length down to H3's `5 + 17 × n` frame grid, potentially omitting up to 16 trailing source frames. **source_start_seconds** chooses where the whole project begins; it does not need to be changed on continuation.

Continuation resumes after a completed save. It does not recover an interrupted sampler halfway through its current queued run. Reduce new_chunks to save more frequently.

<!-- workflow-panel -->
# Segmentation blur and controls

The source feeds a clear encode branch and a separate motion-control branch. Only the control branch receives blur and image noise. The clear source remains available for inpainting and its audio remains available for the soundtrack.

**Enable segmentation blur** is the only on/off control. When false, the Sapiens2 loader, segmentation and mask-extraction branch are not requested. When true, the segmentation selects **Face_Neck + Hair + Eyeglass**. **Upper/Lower Lip, Upper/Lower Teeth and Tongue** form a protection mask. Hands, props and background are excluded when the segmentation labels them correctly.

**Blur Segmented Face and Hair** exposes strength, region growth, protection growth and a minimum component size. Defaults are **strength 0.5**, **region_grow 2**, **protect_grow 2**, and **min_component 0.05**. The component filter removes tiny stray blobs relative to the largest selected component. Lower it if a small secondary subject loses too much mask coverage.

This is the Sapiens2 segmentation path. It uses no detector ellipse, gender filter, face tracking or InsightFace likeness measurement. Segmentation can still make mistakes; inspect the control around occlusions or small faces.

**Control image noise** defaults to **0.10**. Set it to 0 to skip noise. When inpainting is enabled, noise follows its white region; otherwise it covers the full control frame. This value is independent of sampler denoise.

**Optional second aligned control video** starts muted. To use it, select its file and set the node's mode to Always, then refer to it as `<Video 2>` in the prompt. Its time offset and length follow the source automatically. Leave it muted when unused.

For saved RefMods, place files in `models/refmods/`, refresh the loader and select a slot. Start with strength/copies 1, and use the displayed prompt hint. Leave unused slots at `(none)`.

**RefMod only:** mute or bypass **Reference image — Picture 1**. It connects directly to the conditioner's optional image input, so no reference picture or placeholder file is required. Remove `<Picture 1>` from the prompt and describe the desired subject using **RefMod prompt hint**, for example: `A person with short dark hair follows the movement and timing in <Video 1>.` RefMods have no `<Picture N>` tag. `<Video 1>` still refers to the source control. Enable the image loader again to use both.

For an API run without an image, omit the `LoadImage` node and the `ref_images` input on `MMH3ReferenceMultiPrompt`. To use several pictures, connect an `MMH3ImageList` to that input; mute the whole image branch, including the list node, when returning to RefMod only.

<!-- workflow-panel -->
# Inpainting, audio and color

To preserve selected source regions, choose an aligned grayscale video in **Inpaint mask path**, then set **Enable inpainting = true**. This single switch also selects full overlap pinning, as required for source preservation. Turning it off restores five-frame context automatically and skips the mask loader.

**White regenerates; black preserves.** The mask is thresholded at 0.5. It must cover the same source timeline and crop; source and mask are resized to the selected output dimensions. Include hair in the white region if you want it replaced too. The segmentation blur mask controls guidance and is separate from this inpaint mask.

Masks are reduced to H3's latent grid using `max`. A white region can expand to the token cells it touches. Black areas preserve the source latent; normal VAE decoding and MP4 encoding can still change individual pixels. Full-frame regeneration is the default when inpainting is off.

The defaults preserve the source soundtrack: **Audio mask = 0** and **Use source soundtrack? = true**. To generate new audio, set the mask to 1 and the soundtrack switch to false. The accepted prefix's audio stays preserved on a continuation. This workflow expects an audio track in the source video.

The saved continuation pair uses the raw render, keeping accepted pixels and the model's latent appearance aligned. No external grading script is required. For a final color grade, grade the completed full video once; do not replace intermediate paired videos with separately graded copies. This avoids a graded-prefix/raw-continuation color step. It does not correct new color drift generated later by the model.

If a resume fails, check Project name, saved run, accepted chunk count, source start and window grid. If inpainting changes the wrong area, check polarity and alignment. If memory runs out, reduce output size or new_chunks; segmentation can also use a smaller compatible model. The reference conditioning, full accepted prefix and assembled output still occupy memory, so continuation does not make total RAM usage constant.

## Validation scope

Validated with ComfyUI `a20738f1` and fresh public dependency checkouts on 2026-09-14. The release includes graph/schema checks and CPU checks for continuation geometry, a native MP4/master save, accepted video/audio, lazy optional branches, segmentation-only blur and the native sampler path. A full model render of this packaged workflow has not been run.
