# 05A — Start here

## Basic workflow: four steps

1. **Load source video.** Enter its path in **2. Source video path**. The source automatically supplies Control 1 and the original soundtrack. A separate control video or mask is not required.
2. **Write your prompt.** Describe the result in **4. Window prompts — separate with |**. Use `<Video 1>` for source motion. For multiple chunks, enter one prompt per chunk separated by `|`, and set **Number of windows / prompts** to match. The defaults need two prompts.
3. **Add reference images.** Upload the desired subject in **3. Reference image — Picture 1**. Connect more Load Image nodes to **Reference pictures** if needed. Refer to them as `<Picture 1>`, `<Picture 2>`, etc., in connection order.
4. **Define blur.** In **Blur source faces → Control 1**, choose which faces to affect (`gender` / `all_faces`), whether to keep the mouth, and the blur strength and expansion. **enabled = false** turns face blur off. The separate noise node defaults to **0.10**; set its strength to 0 if unwanted.

Then **Queue**. Output saves under `ComfyUI/output/video/HybridAccessible`. For your first run, leave the optional inpaint-mask, Control 2 and hair-segmentation groups muted. The default span is **209 frames / 8.71 seconds** at 24 fps.

**First-time setup:** get the models from the **Model downloads and folders** panel, select them in the model loaders, and restart/refresh ComfyUI after installation.

## Install and select files

Install **ComfyUI-HybridWindows**, **ComfyUI-MMH3Tools**, **ComfyUI-VideoHelperSuite**, [**h3_face_tools**](https://github.com/Jalen-Brunson/h3_face_tools), **ComfyUI-Sapiens2**, and [**ComfyUI-MiniMaxH3Mod**](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod), then restart ComfyUI. Use ComfyUI with native H3 support. The hybrid sampler and its joint-stage helpers come from **ComfyUI-HybridWindows**. MMH3Tools supplies reference conditioning and shared AV/window utilities; no unpublished joint-sampler node is required.

Select the **H3 Ref2VA base**, **8-step DMD Turbo LoRA**, H3 text encoder, video VAE and audio VAE. Defaults use **Euler / simple**, eight steps, six sequential steps and two joint steps, with video/audio sigma shifts 12/3. This differs from the original 05's extension-provided beta57 schedule.

Only the Source video path is required by default:

- **Source video:** the original footage whose unmasked areas and soundtrack you want to keep. This example requires a source audio track.
- **Mask video (optional):** an aligned grayscale video specifying where to regenerate. Its group starts muted.
- **Motion/reference video (optional Control 2):** an additional aligned control. Its group starts muted. **Control 1 automatically uses the source video through the face-blur node**, so a separate control file is not required. Native noise at **0.10** is composited into this control after blur.

The source and any enabled optional loaders share **Start time**, frame count, output dimensions and **24 fps**. Use aligned files covering the entire requested interval. If you pre-trim one file, trim the others identically and set Start time to zero. The workflow resizes the videos; it does not repair different crops or timing.

Load your desired appearance in **Reference image — Picture 1**. The bundled portrait is a placeholder. Copy it from `example_workflows/assets/` to `ComfyUI/input/hybrid_windows/`, or upload your own image. Add other images to **Reference pictures** if needed.

## Run and save

Set the window count and matching prompts, select a source and reference picture, and queue.

To use a mask, unmute all nodes in **OPTIONAL Inpaint mask — enable group to use** (set their mode to Always) and choose a matching mask video. To use a second control, do the same for **OPTIONAL Control 2 — enable group to use**. To disable either feature, mute the whole group again. Muted groups need no valid file path; they are not executed.

**Without an inpaint mask, the whole picture may regenerate.** The source still guides movement through Control 1, but no source-picture regions are pinned. Source audio remains preserved by default. Output saves under `ComfyUI/output/video/HybridAccessible`.

The repo copy has placeholder paths. The local copy preserves your selected paths. Verify that any optional mask/control you enable belongs with the source.

This edition has no automatic resume/master assembly, prompt-file loading, disk source cache, delayed control/mask schedules, audio frame-range editor, or diagnostic branches. RefMod loading and application are included; the trainer/extractor, background removal and custom attention patches are omitted. Face blur and a disabled Sapiens2 hair-segmentation branch are included. No external hair/region-mask video loader is needed. The original 05 remains unchanged.

<!-- workflow-panel -->
# Model downloads and folders

These are the files selected in the shared workflow. Open a link and click **Download**. Keep the filenames below. Folder paths are relative to your **ComfyUI** folder unless stated otherwise; create missing folders, then refresh model lists or restart ComfyUI.

## Required generation models

| Loader | Download | Save inside ComfyUI |
| --- | --- | --- |
| H3 Ref2VA model | [minimax_h3_ref2va_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/diffusion_models/minimax_h3_ref2va_int8_convrot.safetensors) | `models/diffusion_models/` |
| 8-step Turbo LoRA | [minimax_h3_dmd_8step_turbo.safetensors](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI/blob/main/experimental/minimax_h3_dmd_8step_turbo.safetensors) | `models/loras/minimax/` |
| H3 text encoder | [qwen3vl_32b_minimax_h3_int8_convrot.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors) | `models/text_encoders/` |
| Video VAE | [minimax_h3_video_vae_int8_convrot.safetensors](https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main/minimax_h3_video_vae_int8_convrot.safetensors) | `models/vae/` |
| Audio VAE | [minimax_h3_audio_vae_fp32.safetensors](https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main/vae/minimax_h3_audio_vae_fp32.safetensors) | `models/vae/` |

In **H3 text encoder**, choose type **minimax**. Select the Turbo LoRA at strength **1.0**. Use the Ref2VA base with the listed DMD Turbo LoRA; the workflow is set to **8 steps / Euler / simple**, with **6 sequential + 2 joint** steps and video/audio shifts **12 / 3**. Keep the audio VAE selected even when preserving the source soundtrack: this flow encodes source audio for sampling.

## Face blur: InsightFace buffalo_l

Download **buffalo_l.zip** from the [official InsightFace model-package release](https://github.com/deepinsight/insightface/releases/tag/v0.7) and extract the ONNX files.

In **InsightFace model folder**, select the folder containing the extracted ONNX files. The default is `models/insightface/buffalo_l`, relative to your ComfyUI installation:

`ComfyUI/models/insightface/buffalo_l/det_10g.onnx`

Keep the other buffalo_l ONNX files beside it, without another nested buffalo_l folder. You can enter an absolute folder path for models stored elsewhere. The loader's **insightface** output is connected to the source face-blur node; it also works with Face Likeness Probe. No source-code edits or automatic downloads are needed. ONNX sessions load only when the face node uses them, and **unload_after** still releases the cached sessions. Legacy face nodes without a loader use InsightFace's standard `~/.insightface/models/buffalo_l` location.

The ComfyUI Python environment also needs [InsightFace](https://github.com/deepinsight/insightface/tree/master/python-package) and [ONNX Runtime](https://onnxruntime.ai/docs/install/). Match the face node's provider to your installation: **cuda** for GPU execution or **cpu**. Set the node's enabled switch to false to skip face analysis while setting up the rest of the flow.

## Optional face + hair segmentation blur: Sapiens2

Download [sapiens2_5b_seg.safetensors](https://huggingface.co/facebook/sapiens2-seg-5b/blob/main/sapiens2_5b_seg.safetensors) from [Meta's Sapiens2 segmentation model page](https://huggingface.co/facebook/sapiens2-seg-5b).

Save it in `ComfyUI/models/sapiens2/`, then select it in **Optional hair segmentation model**. Choose a **segmentation** checkpoint, not a pose or normals model.

This checkpoint is only needed when you unmute the **OPTIONAL Face + hair seg blur** group. Leave the group muted to skip its model loading and execution. The existing [ComfyUI-Sapiens2 nodes](https://github.com/kijai/ComfyUI-Sapiens2) provide the segmentation and per-class extraction.

**Why this is the better blur.** By default the blur region is an ellipse fitted to the face detector's bounding *box*. An ellipse cannot tell what is inside it, so it also softens anything that happens to sit in *front* of the face — hands, jewellery, props — while clipping the jaw and neck it is meant to cover. The group extracts **Face_Neck + Hair + Eyeglass** and combines them into `region_mask` instead. Because `Upper_Lip`, `Lower_Lip`, `Upper_Teeth`, `Lower_Teeth` and `Tongue` are separate Sapiens2 classes, that union is already "face and hair minus mouth and lips" — lip articulation survives without any geometric mouth cut. `Upper_Lip + Lower_Lip` additionally feed `protect_mask` so the small `region_grow` cannot dilate the face region over the lips.

**Two switches, not one.** Unmute the group **and** set the blur node's **face_detector** to **false**. With `face_detector` on, the ellipse is still drawn *in addition to* the region mask, which is exactly the collateral blur this group avoids. Measured over a 4153-frame clip: residual face recognition 0.10 with the segmentation versus 0.19 for the ellipse, and 3.2% of the blurred pixels landing outside the face and hair versus 18%. With `face_detector` off no InsightFace model is loaded at all unless `measure_residual` is on.

<!-- workflow-panel -->
# Define chunks and prompts

A **chunk/window** is a stretch of video with one prompt. **Overlap** is the shared stretch between neighboring windows. These are frame counts, not sampler steps.

The controls are:

- **Frames per window:** how many frames each window sees; default **124**.
- **Overlap frames:** how many frames neighboring windows share; default **39**.
- **Number of windows / prompts:** how many windows to generate; default **2**. Enter that many pipe-separated prompts.
- **Start time in seconds:** where all three input videos begin; default **0**.

Total frames are calculated automatically:

`total = window + (count - 1) × (window - overlap)`

At the defaults, each new window advances **85 frames**. Two windows produce **209 frames / 8.71 seconds** at 24 fps. Three produce **294 frames / 12.25 seconds**.

For two windows, using frame numbers starting at 1 relative to the loaded interval:

| Window | Frames | Starts at |
| --- | --- | --- |
| 1 | 1–124 | 0 seconds |
| 2 | 86–209 | 3.542 seconds |

Frames **86–124** appear in both windows. The overlap is included once in the final video. A nonzero Start time shifts both windows into the input files by that amount.

Use H3 frame-grid values **5 + 17k** for window and overlap lengths: examples are windows **124, 192, 243** and overlaps **22, 39, 56**. Overlap must be smaller than the window. Keep the linked settings shared between conditioning and sampling.

## Write one prompt per window

In **Window prompts — separate with |**, use:

`The person in <Picture 1> gently turns left, following <Video 1>. | Continue the same shot as the person returns toward the camera, following <Video 1>.`

The `|` separates windows; a newline alone does not. Do not use a literal pipe within a prompt. Match the window-count control to the number of nonempty prompts.

Write continuous action across the overlap. Keep identity, clothing, background, lighting and camera framing consistent unless a change is intentional. Avoid asking each window to restart a complete movement.

`<Picture 1>` is the first connected reference picture; `<Video 1>` is the first connected reference video. Additional inputs follow their connection order. Tags are not automatically rewritten when you remove or reorder inputs.

**sequential_steps = 6** means six early denoising steps per window, followed by a two-step joint finish. It does not mean six chunks. Set it to **8** for the fully sequential baseline with the eight-step schedule.

<!-- workflow-panel -->
# Source blur and optional Control 2

The source video now feeds two separate branches:

- **Clear source → video/audio encode:** supplies the original footage for inpainting and the original soundtrack.
- **Source → Blur source faces → Add Noise (0.10) → masked composite → Control 1:** supplies motion and composition guidance to the reference encoder as `<Video 1>`.

Blur is applied only to the control branch. It does not blur the source being encoded or the audio. A blurred control can reduce visible face-identity cues while preserving coarse movement; it does not guarantee removal of all identity information.

## Face-blur controls

The **Face Anonymize Video** node detects faces automatically. There is **no hair mask or region-mask video loader**. The optional Sapiens2 group supplies `region_mask` and `protect_mask` directly from the clear source when enabled.

- **enabled:** true applies blur; false passes the original source frames through as Control 1.
- **gender:** `any` by default. Select a specific option only when you want detection filtered that way.
- **all_faces:** true by default; false tracks the largest matching face.
- **keep_mouth:** true by default, retaining mouth/jaw detail for articulation. Turn it off to blur the full detected face region. Note that it keeps the whole lower face, not just the lips: jaw, chin and cheeks all stay sharp, which is a large amount of identity. Prefer the segmentation group over `keep_mouth` when you have the model.
- **face_detector:** true by default. Set it to false to blur **only** `region_mask` (minus `protect_mask`) with no ellipse at all — see the segmentation section above. It requires a connected `region_mask`; without one the node stops with an error rather than silently blurring nothing.
- **region_min_component:** drops `region_mask` blobs smaller than this fraction of the largest one, before `region_grow` inflates them, clearing stray segmentation specks on the background or on props. 0 disables it; 0.05 is a good default with the segmentation group, and still keeps a second performer's face.
- **mouth_line:** controls how far down the face the upper-face blur reaches when keep_mouth is on; default 0.73.
- **strength:** default 0.5; larger values apply stronger blur.
- **expand:** grows the detected face box; default 0.45.
- **hold / ema:** retain and smooth the tracked box when detection varies between frames.
- **provider:** CUDA by default; CPU is available when your ONNX Runtime setup requires it.

`hair_expand` stays 0, `output_mask` stays false, and residual measurement is off by default. The mask used for inpainting is independent of face detection and the optional hair segmentation.

Face blur requires [**h3_face_tools**](https://github.com/Jalen-Brunson/h3_face_tools), **InsightFace with buffalo_l models**, and an appropriate **ONNX Runtime** installation. Setting enabled to false skips face analysis while leaving the source connected as Control 1.

## Source-control noise — 0.10

**Image Add Noise** adds noise at strength **0.10** to the blurred source-control images. **Image Composite Masked** uses the blurred image as its destination and the noisy image as its source.

When the inpaint-mask group is enabled, its white regions receive the noisy control image; black regions retain the blurred control without added noise. When the mask group is muted, the composite has no mask and uses the noisy control across the entire image.

This is an image-noise strength, not sampler denoise. Set noise strength to 0 to disable added noise. It does not change the clear source used for encoding, or the source soundtrack.

## Optional Sapiens2 hair region

The existing **ComfyUI-Sapiens2** package provides **Sapiens2 Loader → Body-Part Segmentation → Seg Extract Class**. The extractor is set to **Hair**, with 29 source classes, and its MASK connects to the blur node's `region_mask`.

The entire **OPTIONAL Hair region mask** group starts muted so no segmentation checkpoint is loaded or executed. To use it, select a Sapiens2 **segmentation** checkpoint in its loader and unmute all three nodes. A pose or normals checkpoint cannot be used here. The included selection is `sapiens2_5b_seg.safetensors`; choose another compatible segmentation model if needed.

Segmentation reads the clear source, preserving frame alignment automatically. `frames_per_batch = 1` limits the number of frames processed per model call. The model can still require substantial VRAM.

With the group enabled, **region_mode = blur** blurs the segmented hair as well as the detected face. Other region modes on the face node change how that region is replaced; select them deliberately. `region_grow` expands the selected region, and mouth_keepout can protect the mouth. Turning face blur's enabled switch off skips its effect, but mute the hair group too when you want to skip segmentation work.

The hair mask controls preprocessing of **Control 1**. It does not automatically become the sampler's inpaint mask. To replace hair in the final video while an inpaint mask is enabled, that inpaint mask must also allow the hair area to regenerate.

## Optional second motion control

Control 1 is always the source-derived video. For extra guidance, unmute the entire **OPTIONAL Control 2** group, choose its video path, and mention `<Video 2>` in the relevant prompts. The second input uses the same start time, frame count and output dimensions as the source. Leave the group muted when not needed; a missing file in that muted branch does not block a run.

Remove `<Video 2>` references from prompts when Control 2 is disabled. `<Video 1>` continues to refer to the source-derived control.

For a short static likeness clip instead of an aligned second control, list `ref_video_1` in **static_ref_videos**. That socket is the second video input. Its loader still uses the linked start time and frame cap; adjust those if the clip needs a different interval. Leave static_ref_videos empty for normal time-aligned motion controls.

## Blur is not an inpaint mask

Blur changes the guidance H3 receives. A white inpaint mask permits source-picture regions to regenerate; black pins the encoded source there. The reference picture supplies the desired appearance. A black inpaint region will not be replaced simply because Control 1 is blurred. With the mask group disabled, the entire picture is free to regenerate.

<!-- workflow-panel -->
# Inpaint mask, audio and output

## Enable and build the inpaint mask

The mask group starts muted. Leave it muted for full-picture generation guided by the source. To preserve selected source regions, unmute all nodes in **OPTIONAL Inpaint mask**, then choose the mask file.

Prepare a grayscale mask video aligned with the Source video, with one mask frame per source frame:

- **White = regenerate** this area using the prompts and references.
- **Black = preserve** this area of the encoded source.

For a face replacement, white should cover the intended face region through its movement. Include hair or other features only if you want them regenerated too. Keep the background black when you want it preserved. A moving subject needs a moving mask; a fixed white box only covers the same screen position.

The loader reads the mask's **red channel**, then **Binary inpaint mask** thresholds it at **0.5**. Values above 0.5 become white; values at or below it become black. Use ordinary black/white RGB content rather than relying on a file's alpha channel. Gray feathering is discarded by this threshold.

Source and mask must use the same crop and timing. Their original resolutions may differ if the mask describes the same full-frame composition; both are resized to the linked output dimensions. Check alignment after resizing. Compression artifacts around the threshold can change which pixels regenerate.

The sampler reduces masks to H3's spatial and temporal token grid. With **denoise_mask_mode = max**, a white area can expand to the token cells it touches. Very thin mask details and exact pixel boundaries are not preserved. Black areas preserve the source latent, not necessarily bit-identical source pixels after VAE decoding.

An all-white mask allows the full picture to regenerate. An all-black mask preserves the encoded source picture. Blur is not a substitute for this mask.

## Audio choices

Defaults preserve the source soundtrack in two places:

- **Audio: 0 preserves / 1 regenerates = 0:** keeps the encoded source audio fixed during sampling.
- **Original soundtrack? = true:** sends the loaded source audio directly to Create Video, avoiding an audio VAE round trip. Saving MP4 may still re-encode the audio.

To generate new audio, set the audio mask to **1** and the soundtrack switch to **false**. If the switch stays true, the saved video still uses the source soundtrack even if new audio was generated.

This edition uses a whole-clip audio setting. It does not provide selective audio regeneration by frame range. The default source path must provide audio; silent-video setup requires changing that input path through the graph.

## Color and useful checks

**Optional color stabilization** acts after decoding and may adjust colors outside the inpaint region too. Disable it when checking source preservation or when the scene intentionally changes color or lighting.

If the intended region stays unchanged, check that its mask is white. If too much changes, check mask polarity, alignment and token-grid expansion. If appearance drifts, check the picture/video tags and whether the control still contains competing appearance detail.

If a run reports a conditioning/window-count mismatch, count the pipe-separated prompts and compare them with Number of windows / prompts. If a loader returns too few frames, shorten the requested interval or provide longer aligned input files.

<!-- workflow-panel -->

# Optional RefMods — load saved mods

The four-step basic workflow still works with every RefMod slot set to **(none)**. Reference images remain available as usual. No trainer or extraction node is included.

## Install and choose a mod

Install [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod) for **Load H3 RefMods**, and use the current ComfyUI-MMH3Tools for **Apply H3 RefMods to Cond Set**. This branch does not require the VLM node pack.

Place saved H3 RefMod `.safetensors` files in `ComfyUI/models/refmods/`, then refresh ComfyUI so the loader dropdown sees them. Older mods with a matching `.json` sidecar need that file beside the weights. These are saved RefMods, not ordinary LoRAs. The extension also includes an example in its `mods/` folder; see its [examples and usage](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod#readme).

1. In **Optional RefMods — select saved mods**, choose a file in **mod_1**. Leave unused slots at **(none)**.
2. Start with **strength_1 = 1** and **copies_1 = 1**. Reduce strength for less influence. Extra copies increase the reference-token workload and memory use.
3. Read **RefMod prompt hint** after execution and describe the intended subject or concept in your window prompts. RefMods do not receive a `<Picture N>` tag. Keep those tags for the connected reference images.
4. **Apply RefMods to every chunk** inserts the selected mods into each chunk's conditioning. **retention** multiplies all loader strengths; zero disables their effect. Keep **insert_position = before_controls** and **controls_override = -1** so source Control 1 and optional Control 2 retain their positions.

To skip RefMods, set every loader slot to **(none)**. The loader then needs no mod file and the Apply node passes the original conditioning through unchanged. You can also set individual strengths to zero. Do not mute the Apply node: it carries the conditioning connection to the sampler.
