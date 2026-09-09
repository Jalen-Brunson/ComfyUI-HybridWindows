"""Rebuild the example graph against this ComfyUI's node schemas (CPU only)."""

import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY = ROOT.parents[1]
sys.path.insert(0, str(COMFY))
sys.argv = [sys.argv[0], "--cpu"]
import comfy.options
comfy.options.enable_args_parsing()
import nodes
from comfy_extras import nodes_audio, nodes_minimax_h3, nodes_primitive, nodes_video


def example_prompts(mode, end_guide=False):
    prompts = {
        'ref2va': [
            (
                'A single uninterrupted studio portrait take of the man in <Picture 1>. The fixed camera tightly '
                'frames his head and shoulders against a plain white wall. His face stays the same size and '
                'position in the frame, with the top of his hair near the upper edge and only his shoulders and '
                'black shirt collar at the bottom. He has short light brown hair brushed forward into a fringe, a '
                'moustache and short goatee. Soft daylight stays constant. He gently glances a little to his '
                'left, then slowly returns his eyes toward the lens. Subtle natural blinking and breathing. The '
                'camera remains completely still for the entire take.'
            ),
            (
                'Continue the same uninterrupted studio portrait take of the man in <Picture 1>. Maintain exactly '
                'the established tight head-and-shoulders composition, fixed camera position, focal length, face '
                'size, plain white wall and soft daylight. His short light brown fringe, moustache, short goatee '
                'and black shirt collar stay unchanged. From his current pose he slowly glances a little to his '
                'right and settles, with subtle natural blinking and breathing. Only his eyes and a small gentle '
                'head turn move within the established framing. The camera remains completely still.'
            ),
            (
                'Continue the same uninterrupted studio portrait take of the man in <Picture 1>. Maintain exactly '
                'the established tight head-and-shoulders composition, fixed camera position, focal length, face '
                'size, plain white wall and soft daylight. His short light brown fringe, moustache, short goatee '
                'and black shirt collar stay unchanged. From his current pose he gently turns back toward the '
                'lens and holds a relaxed forward gaze, blinking naturally and breathing softly. The camera '
                'remains completely still through the end of the take.'
            ),
        ],
        'fl2va': [
            (
                'One continuous portrait shot of a man seated at a white desk in a bright office. He wears the '
                'same black long-sleeved button-up shirt, with short black hair swept to one side and a '
                'clean-shaven face. His hands rest together on the desk. He looks toward the camera, blinking '
                'naturally with a subtle relaxed smile. The camera slowly shifts from a slight angle to a frontal '
                'view. Soft daylight, steady exposure, natural skin texture and color. No cuts, no sudden motion.'
            ),
            (
                'Continue the same uninterrupted portrait shot in the established frontal composition. The camera '
                'is now stationary, preserving the established distance and focal length. The same clean-shaven '
                'man with short side-swept black hair and a black long-sleeved button-up shirt remains seated at '
                'the white desk. His hands rest together. He looks toward the lens with a calm expression, '
                'natural blinking and subtle breathing, then gives a very small relaxed smile. Keep the same soft '
                'daylight, exposure, face size and background throughout.'
            ),
            (
                'Continue the same uninterrupted portrait shot in the established frontal composition. The camera '
                'remains stationary at the same distance and focal length throughout the shot. The same '
                'clean-shaven man with short side-swept black hair and a black long-sleeved button-up shirt '
                'remains seated at the white desk, hands together. He maintains a calm gaze toward the lens, '
                'blinking naturally and breathing softly, and holds a comfortable pose as the take ends. Keep the '
                'same face size, background, soft daylight and exposure.'
            ),
        ],
        'fl2va_end': [
            (
                'One continuous portrait shot of a man seated at a white desk in a bright office. He wears the '
                'same black long-sleeved button-up shirt, with short black hair swept to one side and a '
                'clean-shaven face. His hands rest together on the desk. He looks toward the camera, blinking '
                'naturally with a subtle relaxed smile. The camera slowly shifts from a slight angle to a frontal '
                'view. Soft daylight, steady exposure, natural skin texture and color. No cuts, no sudden motion.'
            ),
            (
                'Continue the same uninterrupted portrait shot in the established frontal composition. The camera '
                'is now stationary, at the same distance and focal length as the guide image. The same '
                'clean-shaven man with short side-swept black hair and a black long-sleeved button-up shirt '
                'remains seated at the white desk. His hands rest together. He looks toward the lens with a calm '
                'expression, natural blinking and subtle breathing, then gives a very small relaxed smile. Keep '
                'the same soft daylight, exposure, face size and background throughout.'
            ),
            (
                'Continue the same uninterrupted portrait shot in the established frontal composition. The camera '
                'remains stationary at the same distance and focal length as the guide image. The same '
                'clean-shaven man with short side-swept black hair and a black long-sleeved button-up shirt '
                'remains seated at the white desk, hands together. He maintains a calm gaze toward the lens, '
                'blinking naturally and breathing softly, and settles into the final guide image pose. Keep the '
                'same face size, background, soft daylight and exposure.'
            ),
        ],
    }
    return prompts["fl2va_end" if mode == "fl2va" and end_guide else mode]

def workflow_notes(mode, end_guide=False):
    variant = "FL2VA" if mode == "fl2va" else "Ref2VA"
    size = "480 x 832" if mode == "fl2va" else "832 x 480"
    official = "https://huggingface.co/Comfy-Org/MiniMax-H3/blob/main"
    kijai = "https://huggingface.co/Kijai/MiniMax-H3-experimental/blob/main"
    base = f"minimax_h3_{mode}_int8_convrot.safetensors"
    pdd = f"MiniMax-H3-{variant}-Acc-8Step_comfy.safetensors"
    encoder = "qwen3vl_32b_minimax_h3_int8_convrot.safetensors"
    video_vae = "minimax_h3_video_vae_int8_convrot.safetensors"
    audio_vae = "minimax_h3_audio_vae_fp32.safetensors"
    images = (
        "Use the bundled `fl2va_stock_first.png` and `fl2va_stock_last.png`, "
        "or upload your own first and last images in the two **Load Image** nodes. "
        "The first image connects to window 1. The last image also acts as a recurring composition guide "
        "and connects to `last_frame` on all three encoders. Window 1 reaches that view; windows 2 and 3 "
        "continue the shot from it. This example favors a stable long shot: the ending composition is "
        "reached before the end of the whole video. For first-image-only generation, "
        "disconnect `last_frame` on all three encoders."
        if mode == "fl2va" else
        "Use the bundled `ref2va_stock_portrait.jpg`, or upload your own image in "
        "**Load reference image**. It is already connected to all "
        "three conditioning nodes. Refer to it as `<Picture 1>` in your prompts. "
        "It guides appearance throughout the video rather than fixing the opening pose."
    )
    if mode == "fl2va" and not end_guide:
        images = ("Load your **starting image** in Load FIRST frame. That is the only image used by default. "
                  "The optional ending-image node is disconnected. For a stable target composition, open "
                  "the **FL2VA - recurring end guide** variant, which reaches that view in the first window "
                  "and uses it to guide each continuation. A last image connected only to the final window "
                  "can cause a late reframe; that is not the recommended long-shot setup.")
    extend = (
        "Keep the first image only on window 1. Connect the recurring last-image guide to the added "
        "encoder and describe a continuation of the established shot."
        if mode == "fl2va" else
        "Connect the same reference image to each added conditioning node."
    )
    if mode == "fl2va" and not end_guide:
        extend = "Keep the starting image only on window 1; added windows continue through overlap carry."
    label = "FL2VA" if mode == "fl2va" else "R2V (Ref2VA)"
    return f"""# {label} Hybrid - start here

Build one continuous video from three overlapping prompt windows. The first sampler works through their early steps in order; the second finishes them together, sharing predictions in the overlap.

## Install and download

Install [ComfyUI-HybridWindows](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows) in `ComfyUI/custom_nodes/`, use a current ComfyUI with H3 support, then restart. The links below open the exact model pages; click **Download**. Folders are relative to your ComfyUI installation.

- **{variant} base:** [{base}]({official}/diffusion_models/{base})
  Save in `models/diffusion_models/`.
- **Matching native PDD LoRA:** [{pdd}]({kijai}/loras/{pdd})
  Save in `models/loras/minimax/`.
- **Text encoder:** [{encoder}]({official}/text_encoders/{encoder})
  Save in `models/text_encoders/`; CLIP Loader type = `minimax`.
- **Video VAE:** [{video_vae}]({kijai}/{video_vae})
  Save in `models/vae/`.
- **Audio VAE:** [{audio_vae}]({official}/vae/{audio_vae})
  Save in `models/vae/`.

Refresh the model lists after downloading and select your files in the loaders. Both examples use image conditioning and prompts. The sigma-shift MODEL connects directly to Hybrid Windows.

Use the listed base with its matching PDD LoRA at **1.0**. A model with PDD already baked in would apply the same changes twice. The Ref2VA and FL2VA base/LoRA pairs are different.

## Run the flow

1. Copy the bundled images from [example_workflows/assets](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/tree/main/example_workflows/assets) into `ComfyUI/input/hybrid_windows/`. The assets folder includes source credits and license links. You can also upload them through Load Image and select the uploaded filename.
2. {images}
3. Edit **Prompt 1, 2 and 3** as an opening, continuation and ending. Keep identity, clothing, hairstyle, camera distance and lighting consistent. Describe only that window's part of the action; repeating a complete camera move in every box can ask the model to restart it.
4. Set **Width / Height** (default **{size}**). **Window frames = 243**, **overlap = 39** gives **651 frames / 27.125 seconds** total at 24 fps. The full latent length is calculated automatically.
5. Queue once; the complete video and generated audio save under `ComfyUI/output/video/HybridNative_{variant}*`.

## Optional color stabilization

**Video Color Stabilize** sits after Decode video and before Create Video. It is enabled at **strength 0.85**, uses the generated **1-8 second** interval as its color reference, and smooths tint/saturation drift over **2 seconds**. Brightness is preserved. Turn **enabled** off (or set strength to zero) for exact frame passthrough. The shared **Output FPS** node drives both the correction and video creation.

The optional `source_frames` input is for original, unnoised video frames aligned one-for-one with the complete generated timeline. It accounts for source color changes relative to the opening. Leave it disconnected for these image-conditioned examples; do not connect the portrait or first/last guide images. Without a source, intentional color changes may also be reduced. Process each continuous shot as one batch; split at scene cuts. This changes decoded color only, not sampling, audio, sharpness or spatial mottling.

## Keep the two samplers paired

- Both samplers: **8 total steps, Euler, simple, CFG 1**. Video/audio sigma shifts: **12 / 3**.
- **Switch at step = 6** connects both sides of the handoff. Values **1-7** use a joint finish; **8** is a fully sequential baseline. **Total steps** also connects to Hybrid Windows.
- **Sequential warmup:** start 0, end 6; add noise **enabled**; return leftover noise **enabled**.
- **Joint finish:** start 6, end 8; add noise **disabled**; return leftover noise **disabled**. Its latent comes from the first sampler.
- For a new generation, set the same new noise seed in both samplers.

For the **8+0 baseline**, set only **Switch at step = 8**. Each window finishes all eight steps before passing its completed overlap forward. The second native sampler passes the latent through without sampling. This is useful for checking whether a visible change comes from joint finishing; it does not blend or repair decoded frames.

## Longer videos

Window/overlap lengths follow H3's **17k+5** frame grid (overlap examples: 22, 39, 56). Overlap must be smaller than the window. Total frames = window + (number of prompts - 1) x (window - overlap).

To add a window, duplicate a prompt box and its native H3 conditioning node. Connect its positive output to the next Hybrid Windows prompt input, and share the width, height, window-length, CLIP and VAE connections. {extend}

Each queue generates a fresh full timeline. [More detail and sampler diagram](https://github.com/Jalen-Brunson/ComfyUI-HybridWindows#how-it-works).

## Research credit

Credit to **David Ruhe, Jonathan Heek, Tim Salimans, and Emiel Hoogeboom** for
[Rolling Diffusion Models](https://proceedings.mlr.press/v235/ruhe24a.html)
(ICML 2024), an inspiration for temporal generation through sliding-window
denoising. This H3 adaptation uses sequential Euler warmup followed by a joint
window finish.
"""


async def build(mode="ref2va", end_guide=False):
    assert mode in ("ref2va", "fl2va")
    fl2va = mode == "fl2va"
    assert not end_guide or fl2va
    for module in (nodes_audio, nodes_minimax_h3, nodes_primitive, nodes_video):
        extension = await module.comfy_entrypoint()
        for cls in await extension.get_node_list():
            nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls
    assert await nodes.load_custom_node(str(ROOT))
    graph_nodes, links, api = [], [], {}

    def add(kind, title, xy, size, values=None):
        cls = nodes.NODE_CLASS_MAPPINGS[kind]
        if hasattr(cls, "GET_NODE_INFO_V1"):
            info = cls.GET_NODE_INFO_V1()
        else:
            info = {"input": cls.INPUT_TYPES(), "output": cls.RETURN_TYPES,
                    "output_name": getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)}
        node_id = len(graph_nodes)+1
        values = values or {}
        sockets, widgets, named = [], [], {}

        def fields(inputs, prefix=""):
            for section in ("required", "optional"):
                for short, definition in inputs.get(section, {}).items():
                    typ = definition[0]
                    options = definition[1] if len(definition) > 1 else {}
                    name = prefix + short
                    if typ == "COMFY_AUTOGROW_V3":
                        template = options["template"]
                        members = [n for n in values if n.startswith(name+".")]
                        input_type = next(iter(template["input"]["required"].values()))[0]
                        for member in members:
                            sockets.append({"name": member, "type": input_type, "link": None, "shape": 7})
                        continue
                    socket_type = "COMBO" if isinstance(typ, list) else typ
                    is_widget = isinstance(typ, list) or typ in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3")
                    socket = {"name": name, "type": socket_type, "link": None}
                    if section == "optional":
                        socket["shape"] = 7
                    if is_widget:
                        socket["widget"] = {"name": name}
                        choices = typ if isinstance(typ, list) else options.get("options", [])
                        default = options.get("default", choices[0] if choices else
                                              (False if typ == "BOOLEAN" else "" if typ == "STRING" else 0))
                        value = values.get(name, default)
                        if isinstance(value, tuple):
                            value = default
                        if isinstance(value, dict):
                            value = value["key"]
                        widgets.append(value)
                        named[name] = value
                        if options.get("control_after_generate"):
                            widgets.append("fixed")
                        sockets.append(socket)
                        if typ == "COMFY_DYNAMICCOMBO_V3":
                            choice = next(c for c in choices if c["key"] == value)
                            fields(choice["inputs"], name+".")
                    else:
                        sockets.append(socket)
        fields(info["input"])
        known_names = {s["name"] for s in sockets}
        assert not set(values)-known_names, (kind, set(values)-known_names)
        # Match Comfy's node construction: connection sockets precede widgets.
        sockets.sort(key=lambda x: "widget" in x)
        outputs = [{"name": label, "type": typ, "links": []}
                   for label, typ in zip(info["output_name"], info["output"])]
        node = {"id": node_id, "type": kind, "pos": list(xy), "size": list(size),
                "flags": {}, "order": node_id-1, "mode": 0, "inputs": sockets,
                "outputs": outputs, "title": title,
                "properties": {"Node name for S&R": kind},
                "widgets_values": widgets, "widgets_values_named": named}
        if kind not in ("H3HybridWindows", "H3HybridControlNet", "VideoColorStabilize"):
            node["properties"]["cnr_id"] = "comfy-core"
        graph_nodes.append(node)
        prompt_inputs = dict(named)
        for name, value in values.items():
            if isinstance(value, tuple):
                origin, output = value
                origin_node = graph_nodes[origin-1]
                slot = next(i for i, s in enumerate(sockets) if s["name"] == name)
                link_id = len(links)+1
                typ = origin_node["outputs"][output]["type"]
                links.append([link_id, origin, output, node_id, slot, typ])
                sockets[slot]["link"] = link_id
                origin_node["outputs"][output]["links"].append(link_id)
                prompt_inputs[name] = [str(origin), output]
        api[str(node_id)] = {"class_type": kind, "inputs": prompt_inputs, "_meta": {"title": title}}
        return node_id

    ref = add("LoadImage", "Load FIRST frame - start of video" if fl2va else "Load reference image - all three windows",
              (540, 100), (400, 400),
              {"image": "hybrid_windows/fl2va_stock_first.png" if fl2va else "hybrid_windows/ref2va_stock_portrait.jpg"})
    if fl2va:
        last = add("LoadImage", "Load LAST frame - recurring composition" if end_guide else "Optional ending image - disconnected", (540, 650), (400, 400),
                   {"image": "hybrid_windows/fl2va_stock_last.png"})
    base = add("UNETLoader", f"H3 {'FL2VA' if fl2va else 'Ref2VA'} base", (40, 100), (440, 140),
               {"unet_name": f"minimax_h3_{mode}_int8_convrot.safetensors", "weight_dtype": "default"})
    lora = add("LoraLoaderModelOnly", "Native PDD LoRA", (40, 290), (440, 150),
               {"model": (base, 0), "lora_name": f"minimax/MiniMax-H3-{'FL2VA' if fl2va else 'Ref2VA'}-Acc-8Step_comfy.safetensors", "strength_model": 1.0})
    shift = add("MiniMaxH3SigmaShift", "Native video/audio shifts", (40, 490), (440, 150),
                {"model": (lora, 0), "shift_video": 12., "shift_audio": 3.})
    clip = add("CLIPLoader", "H3 text encoder", (40, 690), (440, 150),
               {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"})
    vae = add("VAELoader", "Video VAE", (40, 890), (440, 100), {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"})
    audio_vae = add("VAELoader", "Audio VAE", (40, 1040), (440, 100), {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"})
    width = add("PrimitiveInt", "Width", (1040, 1100), (220, 110), {"value": 480 if fl2va else 832})
    height = add("PrimitiveInt", "Height", (1300, 1100), (220, 110), {"value": 832 if fl2va else 480})
    window = add("PrimitiveInt", "Window frames", (1560, 1100), (220, 110), {"value": 243})
    split = add("PrimitiveInt", "Switch at step", (1820, 1100), (220, 110), {"value": 6})
    steps = add("PrimitiveInt", "Total steps", (2080, 1100), (220, 110), {"value": 8})
    prompt_texts = example_prompts(mode, end_guide)
    prompts = []
    for i in range(3):
        x = 1040 + i*450
        text = add("PrimitiveStringMultiline", f"Prompt {i+1}", (x, 100), (410, 300), {"value": prompt_texts[i]})
        values = {"clip": (clip, 0), "vae": (vae, 0), "prompt": (text, 0),
                  "width": (width, 0), "height": (height, 0), "length": (window, 0)}
        if fl2va:
            if i == 0:
                values["first_frame"] = (ref, 0)
            if end_guide:
                values["last_frame"] = (last, 0)
            kind = "MiniMaxH3ImageToVideo"
            title = (("Window 1 - move into ending view", "Window 2 - continue the shot", "Window 3 - finish the shot")
                     if end_guide else ("Window 1 - starting image", "Window 2 - continuation", "Window 3 - continuation"))[i]
        else:
            values.update({"ref_image_size": "match", "ref_images.ref_image_0": (ref, 0)})
            kind, title = "MiniMaxH3ReferenceToVideo", f"Native H3 conditioning {i+1}"
        encoded = add(kind, title, (x, 460), (410, 470), values)
        prompts.append(encoded)
    setup = add("H3HybridWindows", "Hybrid windows", (2440, 100), (390, 320),
                {"model": (shift, 0), "window_frames": (window, 0), "overlap_frames": 39,
                 "total_steps": (steps, 0),
                 **{f"prompts.positive_{i}": (p, 0) for i, p in enumerate(prompts)}})
    empty = add("EmptyMiniMaxH3LatentAV", "Full timeline latent", (2870, 100), (360, 180),
                {"width": (width, 0), "height": (height, 0), "length": (setup, 3)})
    negative = add("ConditioningZeroOut", "Unused at CFG 1", (2870, 350), (360, 100), {"conditioning": (setup, 2)})
    common = {"noise_seed": 123, "steps": (steps, 0), "cfg": 1., "sampler_name": "euler", "scheduler": "simple",
              "positive": (setup, 2), "negative": (negative, 0)}
    warmup = add("KSamplerAdvanced", "1. Sequential warmup", (2440, 580), (390, 600),
                 {**common, "model": (setup, 0), "latent_image": (empty, 0), "add_noise": "enable",
                  "start_at_step": 0, "end_at_step": (split, 0), "return_with_leftover_noise": "enable"})
    finish = add("KSamplerAdvanced", "2. Joint finish", (2870, 580), (360, 600),
                 {**common, "model": (setup, 1), "latent_image": (warmup, 0), "add_noise": "disable",
                  "start_at_step": (split, 0), "end_at_step": (steps, 0), "return_with_leftover_noise": "disable"})
    fps = add("PrimitiveFloat", "Output FPS", (3360, 950), (420, 110), {"value": 24.})
    decoded = add("VAEDecode", "Decode video", (3360, 100), (420, 100), {"samples": (finish, 0), "vae": (vae, 0)})
    decoded_audio = add("VAEDecodeAudio", "Decode audio", (3360, 260), (420, 100),
                        {"samples": (finish, 0), "vae": (audio_vae, 0)})
    color = add("VideoColorStabilize", "Optional color stabilization", (3360, 450), (420, 400),
                {"images": (decoded, 0), "enabled": True, "strength": .85, "fps": (fps, 0),
                 "reference_start_seconds": 1., "reference_end_seconds": 8., "smoothing_seconds": 2.})
    video = add("CreateVideo", "Create video", (3880, 100), (420, 210),
                {"images": (color, 0), "audio": (decoded_audio, 0), "fps": (fps, 0), "bit_depth": 8, "color_space": "sRGB"})
    add("SaveVideo", "Save hybrid test", (3880, 450), (420, 510),
        {"video": (video, 0), "filename_prefix": "video/HybridNative_FL2VA_EndGuide" if end_guide else "video/HybridNative_FL2VA" if fl2va else "video/HybridNative_Ref2VA", "format": "mp4", "format.codec": "h264",
         "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 16., "codec": "auto"})
    groups = [
        ("Models", [0, 0, 510, 1240]), ("First and last images" if fl2va else "Load reference image", [520, 0, 450, 1240]),
        ("Three prompts", [1000, 0, 1380, 980]), ("Size and handoff", [1000, 1020, 1380, 240]),
        ("Two native samplers", [2400, 0, 870, 1280]), ("Decode, optional color and output", [3320, 0, 1020, 1280]),
    ]
    # Leave a full column for the notes without changing the layout within groups.
    for node in graph_nodes:
        node["pos"][0] += 1160
    for _, bounds in groups:
        bounds[0] += 1160
    note = workflow_notes(mode, end_guide)
    graph_nodes.append({
        "id": len(graph_nodes)+1, "type": "MarkdownNote", "pos": [40, 100], "size": [1040, 2320],
        "flags": {}, "order": len(graph_nodes), "mode": 0, "inputs": [], "outputs": [],
        "title": "READ ME - FL2VA" if fl2va else "READ ME - R2V (Ref2VA)", "properties": {},
        "widgets_values": [note], "widgets_values_named": {"text": note},
        "color": "#234", "bgcolor": "#345",
    })
    groups.append(("Start here - downloads and instructions", [0, 0, 1120, 2480]))
    graph = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/{mode}{'/end-guide' if end_guide else ''}")),
             "revision": 0, "last_node_id": len(graph_nodes),
             "last_link_id": len(links), "nodes": graph_nodes, "links": links,
             "groups": [{"id": i+1, "title": title, "bounding": bounds, "color": "#3f789e", "font_size": 24}
                        for i, (title, bounds) in enumerate(groups)], "config": {}, "version": .4,
             "extra": {"ds": {"scale": .29, "offset": [50, 100]}, "VHS_latentpreview": False}}
    folder = ROOT / "example_workflows"
    folder.mkdir(exist_ok=True)
    name = "H3 Hybrid FL2VA - native KSampler Advanced" if fl2va else "H3 Hybrid R2V - native KSampler Advanced"
    if end_guide:
        name = "H3 Hybrid FL2VA - recurring end guide"
    (folder / f"{name}.json").write_text(json.dumps(graph, indent=2)+"\n")
    (folder / f"{name}.api.json").write_text(json.dumps(api, indent=2)+"\n")
    print(f"Built {mode}: {len(graph_nodes)} nodes / {len(links)} links.")
    return graph, api


if __name__ == "__main__":
    asyncio.run(build())
    asyncio.run(build("fl2va"))
    asyncio.run(build("fl2va", end_guide=True))
