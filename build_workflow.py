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
from comfy_extras import nodes_audio, nodes_images, nodes_minimax_h3, nodes_model_patch, nodes_primitive, nodes_video


async def build(mode="ref2va"):
    assert mode in ("ref2va", "fl2va")
    fl2va = mode == "fl2va"
    for module in (nodes_audio, nodes_images, nodes_minimax_h3, nodes_primitive, nodes_video):
        extension = await module.comfy_entrypoint()
        for cls in await extension.get_node_list():
            nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls
    nodes.NODE_CLASS_MAPPINGS.update(nodes_model_patch.NODE_CLASS_MAPPINGS)
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
        if kind not in ("H3HybridWindows", "H3HybridControlNet"):
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
              {"image": "hybrid_man_first.png" if fl2va else "hybrid_man_reference.png"})
    if fl2va:
        last = add("LoadImage", "Load LAST frame - end of video", (540, 650), (400, 400),
                   {"image": "hybrid_man_last.png"})
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
    width = add("PrimitiveInt", "Width", (1040, 1100), (220, 110), {"value": 832})
    height = add("PrimitiveInt", "Height", (1300, 1100), (220, 110), {"value": 480})
    window = add("PrimitiveInt", "Window frames", (1560, 1100), (220, 110), {"value": 243})
    split = add("PrimitiveInt", "Switch at step", (1820, 1100), (220, 110), {"value": 6})
    steps = add("PrimitiveInt", "Total steps", (2080, 1100), (220, 110), {"value": 8})
    source = add("LoadVideo", "24 fps motion-control video", (40, 1520), (440, 330), {"file": "hybrid_man_loop_90s.mp4"})
    trim = add("Video Slice", "Read first 30 seconds", (540, 1520), (400, 180),
               {"video": (source, 0), "start_time": 0., "duration": 30., "strict_duration": False})
    components = add("GetVideoComponents", "Native control frames", (1020, 1520), (380, 140), {"video": (trim, 0)})
    noise = add("ImageAddNoise", "Control noise", (1460, 1520), (380, 180),
                {"image": (components, 0), "seed": 456, "strength": .1})
    patch = add("ModelPatchLoader", "Native Fun Control model", (1020, 1760), (380, 110),
                {"name": "minimax_h3_fun_controlnet_union.safetensors"})
    control = add("H3HybridControlNet", "Control follows each window", (1900, 1520), (440, 310),
                  {"model": (shift, 0), "model_patch": (patch, 0), "vae": (vae, 0),
                   "control_video": (noise, 0), "strength": .5, "start_percent": 0., "end_percent": 1.})
    prompt_text = ("A continuous locked-camera close-up of the man in <Picture 1>. He looks left and right, "
                   "following the motion-control video. He wears the same black collared shirt throughout. "
                   "His medium-length light brown hair remains swept back with the same side part, and his "
                   "short beard stays unchanged. Bright white studio background, soft even lighting, "
                   "natural skin texture and color, steady framing. No cuts, no camera movement.")
    if fl2va:
        prompt_text = prompt_text.replace("the man in <Picture 1>", "a man")
    prompts = []
    for i in range(3):
        x = 1040 + i*450
        text = add("PrimitiveStringMultiline", f"Prompt {i+1}", (x, 100), (410, 300), {"value": prompt_text})
        values = {"clip": (clip, 0), "vae": (vae, 0), "prompt": (text, 0),
                  "width": (width, 0), "height": (height, 0), "length": (window, 0)}
        if fl2va:
            if i == 0:
                values["first_frame"] = (ref, 0)
            elif i == 2:
                values["last_frame"] = (last, 0)
            kind = "MiniMaxH3ImageToVideo"
            title = ("Window 1 - FIRST frame", "Window 2 - continuation", "Window 3 - LAST frame")[i]
        else:
            values.update({"ref_image_size": "match", "ref_images.ref_image_0": (ref, 0)})
            kind, title = "MiniMaxH3ReferenceToVideo", f"Native H3 conditioning {i+1}"
        encoded = add(kind, title, (x, 460), (410, 470), values)
        prompts.append(encoded)
    setup = add("H3HybridWindows", "Hybrid windows", (2440, 100), (390, 320),
                {"model": (control, 0), "window_frames": (window, 0), "overlap_frames": 39,
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
    decoded = add("VAEDecode", "Decode video", (3360, 100), (420, 100), {"samples": (finish, 0), "vae": (vae, 0)})
    decoded_audio = add("VAEDecodeAudio", "Decode audio", (3360, 260), (420, 100),
                        {"samples": (finish, 0), "vae": (audio_vae, 0)})
    video = add("CreateVideo", "Native video at 24 fps", (3360, 440), (420, 210),
                {"images": (decoded, 0), "audio": (decoded_audio, 0), "fps": 24., "bit_depth": 8, "color_space": "sRGB"})
    add("SaveVideo", "Save hybrid test", (3360, 720), (420, 510),
        {"video": (video, 0), "filename_prefix": "video/HybridNative_FL2VA" if fl2va else "video/HybridNative_Ref2VA", "format": "mp4", "format.codec": "h264",
         "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 16., "codec": "auto"})
    groups = [
        ("Models", [0, 0, 510, 1240]), ("First and last images" if fl2va else "Load reference image", [520, 0, 450, 1240]),
        ("Three prompts", [1000, 0, 1380, 980]), ("Size and handoff", [1000, 1020, 1380, 240]),
        ("Two native samplers", [2400, 0, 870, 1280]), ("Native output", [3320, 0, 500, 1280]),
        ("Motion control - adjust trim duration for longer runs", [0, 1420, 2380, 520]),
    ]
    graph = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, f"https://github.com/Jalen-Brunson/ComfyUI-HybridWindows/{mode}")),
             "revision": 0, "last_node_id": len(graph_nodes),
             "last_link_id": len(links), "nodes": graph_nodes, "links": links,
             "groups": [{"id": i+1, "title": title, "bounding": bounds, "color": "#3f789e", "font_size": 24}
                        for i, (title, bounds) in enumerate(groups)], "config": {}, "version": .4,
             "extra": {"ds": {"scale": .37, "offset": [50, 100]}, "VHS_latentpreview": False}}
    folder = ROOT / "example_workflows"
    folder.mkdir(exist_ok=True)
    name = "H3 Hybrid FL2VA - native KSampler Advanced" if fl2va else "H3 Hybrid - native KSampler Advanced"
    (folder / f"{name}.json").write_text(json.dumps(graph, indent=2)+"\n")
    (folder / f"{name}.api.json").write_text(json.dumps(api, indent=2)+"\n")
    print(f"Built {mode}: {len(graph_nodes)} nodes / {len(links)} links.")
    return graph, api


if __name__ == "__main__":
    asyncio.run(build())
    asyncio.run(build("fl2va"))
