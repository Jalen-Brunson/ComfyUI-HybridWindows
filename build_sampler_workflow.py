"""Build the single-sampler reference-image example using installed node schemas."""

import asyncio
import json
from pathlib import Path

from build_workflow import ROOT, COMFY, nodes
from comfy_extras import nodes_audio, nodes_minimax_h3, nodes_video, nodes_custom_sampler


async def build():
    for module in (nodes_audio, nodes_minimax_h3, nodes_video, nodes_custom_sampler):
        extension = await module.comfy_entrypoint()
        for cls in await extension.get_node_list():
            nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls
    assert await nodes.load_custom_node(str(ROOT.parent / "ComfyUI-MMH3Tools"))
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
        if kind not in ("H3HybridWindows", "H3HybridControlNet", "VideoColorStabilize", "MMH3HybridWindowSampler", "MMH3CondToSet"):
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

    base = add("UNETLoader", "H3 Ref2VA base", (40, 100), (430, 140),
               {"unet_name": "minimax_h3_ref2va_int8_convrot.safetensors", "weight_dtype": "default"})
    lora = add("LoraLoaderModelOnly", "Ref2VA 8-step PDD", (40, 300), (430, 150),
               {"model": (base, 0), "lora_name": "minimax/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors", "strength_model": 1.})
    shift = add("MiniMaxH3SigmaShift", "Video / audio sigma shifts", (40, 510), (430, 150),
                {"model": (lora, 0), "shift_video": 12., "shift_audio": 3.})
    clip = add("CLIPLoader", "H3 text encoder", (40, 720), (430, 150),
               {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"})
    vae = add("VAELoader", "Video VAE", (40, 930), (430, 100),
              {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"})
    audio_vae = add("VAELoader", "Audio VAE", (40, 1090), (430, 100),
                    {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"})
    ref = add("LoadImage", "Reference appearance - Picture 1", (540, 100), (420, 450),
              {"image": "hybrid_windows/ref2va_stock_portrait.jpg"})
    cond = add("MiniMaxH3ReferenceToVideo", "Native reference-image conditioning", (1030, 100), (460, 500),
               {"clip": (clip, 0), "vae": (vae, 0), "width": 832, "height": 480, "length": 243,
                "ref_image_size": "match", "ref_images.ref_image_0": (ref, 0),
                "prompt": "A continuous portrait shot of the person in <Picture 1>. Preserve their appearance and clothing. They gently glance to the side and return their gaze toward the camera, blinking and breathing naturally. Fixed camera, consistent framing and soft daylight throughout. Quiet room ambience."})
    cond_set = add("MMH3CondToSet", "Same reference and prompt for 3 windows", (1030, 680), (460, 140),
                   {"conditioning": (cond, 0), "count": 3})
    empty = add("EmptyMiniMaxH3LatentAV", "Full timeline: 651 frames", (1030, 900), (460, 210),
                {"width": 832, "height": 480, "length": 651})
    noise = add("RandomNoise", "Seed + window index", (1570, 100), (380, 130), {"noise_seed": 123})
    euler = add("KSamplerSelect", "Plain Euler", (1570, 300), (380, 100), {"sampler_name": "euler"})
    sigmas = add("BasicScheduler", "8-step schedule", (1570, 480), (380, 210),
                 {"model": (shift, 0), "scheduler": "simple", "steps": 8, "denoise": 1.})
    sampled = add("MMH3HybridWindowSampler", "6 sequential + 2 joint steps", (2040, 100), (440, 570),
                  {"model": (shift, 0), "noise": (noise, 0), "sampler": (euler, 0), "sigmas": (sigmas, 0),
                   "cond_set": (cond_set, 0), "latent": (empty, 0), "window_frames": 243, "overlap_frames": 39,
                   "sequential_steps": 6, "accumulator_device": "gpu", "denoise_mask_mode": "max",
                   "accepted_prefix_frames": 0, "start_window": 0})
    decoded = add("VAEDecode", "Decode video", (2570, 100), (400, 110), {"samples": (sampled, 0), "vae": (vae, 0)})
    audio = add("VAEDecodeAudio", "Decode audio", (2570, 280), (400, 110), {"samples": (sampled, 0), "vae": (audio_vae, 0)})
    video = add("CreateVideo", "Video + generated audio", (2570, 470), (400, 220),
                {"images": (decoded, 0), "audio": (audio, 0), "fps": 24., "bit_depth": 8, "color_space": "sRGB"})
    add("SaveVideo", "Save reference-image hybrid", (3060, 100), (450, 540),
        {"video": (video, 0), "filename_prefix": "video/HybridSampler_Ref2VA", "format": "mp4",
         "format.codec": "h264", "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 16., "codec": "auto"})
    note = """# Single hybrid sampler — reference image
Install ComfyUI-HybridWindows and ComfyUI-MMH3Tools, then restart ComfyUI. All other processing nodes are native ComfyUI.

Copy example_workflows/assets/ref2va_stock_portrait.jpg into ComfyUI/input/hybrid_windows/, or upload your own image in Load Image. The reference is connected to native H3 Ref2VA conditioning and reused in every window. Edit the prompt to match your image; <Picture 1> refers to it.

Select your Ref2VA base, matching 8-step PDD LoRA, H3 text encoder and video/audio VAEs in the loaders. Model locations and download links are in this repository's README. Use an unmerged base with the LoRA at strength 1.

Defaults: 832 x 480, 651 frames, 24 fps (27.125 seconds). Three windows of 243 frames overlap by 39. Six sequential Euler steps then two joint steps; CFG is 1. Set sequential_steps to 8 for a fully sequential baseline. Do not add Context Windows to the model.

To change duration, keep Cond To Set count and Empty H3 Latent length consistent: total = window + (count - 1) * (window - overlap). Window and overlap use 17k+5 frames. Change width/height in BOTH conditioning and empty latent nodes. The conditioning length is one window.

Queue to save MP4 with generated audio under output/video/HybridSampler_Ref2VA. Hybrid quality remains experimental. This workflow starts fresh; accepted_prefix_frames and start_window stay 0.
"""
    graph_nodes.append({"id": len(graph_nodes)+1, "type": "MarkdownNote", "pos": [540, 650], "size": [420, 980],
                        "flags": {}, "order": len(graph_nodes), "mode": 0, "inputs": [], "outputs": [],
                        "title": "Start here", "properties": {}, "widgets_values": [note]})
    graph = {"version": .4, "revision": 0, "last_node_id": len(graph_nodes), "last_link_id": len(links),
             "nodes": graph_nodes, "links": links, "groups": [], "config": {},
             "extra": {"ds": {"scale": .4, "offset": [30, 50]}}}
    folder = ROOT / "example_workflows"
    name = "H3 Hybrid R2V - single sampler"
    (folder / f"{name}.json").write_text(json.dumps(graph, indent=2) + "\n")
    (folder / f"{name}.api.json").write_text(json.dumps(api, indent=2) + "\n")
    return graph, api


if __name__ == "__main__":
    asyncio.run(build())
