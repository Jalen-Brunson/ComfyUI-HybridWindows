"""Build the accessible fresh-run version of the 05 hybrid workflow."""

import asyncio
import json
import os
import uuid
from pathlib import Path

from build_workflow import ROOT, COMFY, nodes
import server
from comfy_extras import (nodes_audio, nodes_minimax_h3, nodes_video,
                          nodes_custom_sampler, nodes_mask, nodes_primitive,
                          nodes_math, nodes_logic, nodes_preview_any, nodes_images)

NAME = "05A Hybrid - accessible fresh inpaint"
SOURCE = COMFY / "user/default/workflows/H3 Diagnostics/05 Hybrid - preserve accepted prefix.json"


async def build(source_video="input/source.mp4", mask_video_path="input/mask.webm", control_video="input/control.mp4", *, enable_mask=False, enable_control2=False, enable_hair=False):
    server.PromptServer(asyncio.get_running_loop())
    nodes.NODE_CLASS_MAPPINGS.update(nodes_preview_any.NODE_CLASS_MAPPINGS)
    for module in (nodes_audio, nodes_minimax_h3, nodes_video, nodes_custom_sampler,
                   nodes_mask, nodes_primitive, nodes_math, nodes_logic, nodes_images):
        extension = await module.comfy_entrypoint()
        for cls in await extension.get_node_list():
            nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls
    for folder in ("ComfyUI-MMH3Tools", "comfyui-videohelpersuite", "ComfyUI-HybridWindows", "h3_face_tools", "ComfyUI-Sapiens2", "ComfyUI-MiniMaxH3Mod"):
        folder_path = os.environ.get("MMH3TOOLS_TEST_ROOT", str(ROOT.parent / folder)) if folder == "ComfyUI-MMH3Tools" else str(ROOT.parent / folder)
        assert await nodes.load_custom_node(folder_path)
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
        module = cls.__module__
        if module == "nodes" or module.startswith("comfy_extras."):
            node["properties"]["cnr_id"] = "comfy-core"
        elif "MMH3" in module:
            node["properties"]["aux_id"] = "ckinpdx/ComfyUI-MMH3Tools"
        elif "HybridWindows" in module:
            node["properties"]["aux_id"] = "Jalen-Brunson/ComfyUI-HybridWindows"
        elif "ComfyUI-Sapiens2" in module:
            node["properties"]["aux_id"] = "kijai/ComfyUI-Sapiens2"
        elif "ComfyUI-MiniMaxH3Mod" in module:
            node["properties"]["aux_id"] = "Luisacaotica/ComfyUI-MiniMaxH3Mod"
        elif "h3_face_tools" in module:
            node["properties"]["aux_id"] = "Jalen-Brunson/h3_face_tools"
        elif "videohelpersuite" in module:
            node["properties"]["cnr_id"] = "comfyui-videohelpersuite"
            # VHS restores widgets by name, including its frontend preview widget.
            node["widgets_values"] = {**named, "videopreview": {
                "hidden": True, "paused": True, "params": {},
            }}
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
        matched = {graph_nodes[value[0]-1]["outputs"][value[1]]["type"]
                   for name, value in values.items() if isinstance(value, tuple)
                   and next(s for s in sockets if s["name"] == name)["type"] == "COMFY_MATCHTYPE_V3"}
        if matched:
            assert len(matched) == 1, (kind, matched)
            for socket in sockets + outputs:
                if socket["type"] == "COMFY_MATCHTYPE_V3":
                    socket["type"] = next(iter(matched))
        api[str(node_id)] = {"class_type": kind, "inputs": prompt_inputs, "_meta": {"title": title}}
        return node_id

    base = add("UNETLoader", "1. H3 Ref2VA model", (40, 100), (440, 140),
               {"unet_name": "minimax_h3_ref2va_int8_convrot.safetensors", "weight_dtype": "default"})
    lora = add("LoraLoaderModelOnly", "8-step Turbo LoRA", (40, 300), (440, 150),
               {"lora_name": "minimax/minimax_h3_dmd_8step_turbo.safetensors", "strength_model": 1., "model": (base, 0)})
    shift = add("MiniMaxH3SigmaShift", "Video / audio shifts", (40, 510), (440, 150),
                {"model": (lora, 0), "shift_video": 12., "shift_audio": 3.})
    clip = add("CLIPLoader", "H3 text encoder", (40, 720), (440, 150), {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"})
    vae = add("VAELoader", "Video VAE", (40, 930), (440, 100), {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"})
    avae = add("VAELoader", "Audio VAE", (40, 1100), (440, 100), {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"})
    width = add("PrimitiveInt", "Output width", (550, 100), (260, 110), {"value": 832})
    height = add("PrimitiveInt", "Output height", (860, 100), (260, 110), {"value": 480})
    window = add("PrimitiveInt", "Frames per window", (550, 300), (260, 110), {"value": 124})
    overlap = add("PrimitiveInt", "Overlap frames", (860, 300), (260, 110), {"value": 39})
    count = add("PrimitiveInt", "Number of windows / prompts", (550, 500), (260, 110), {"value": 2})
    total = add("ComfyMathExpression", "Total frames (automatic)", (860, 500), (320, 170),
                {"expression": "a + (b - 1) * (a - c)", "values.a": (window, 0),
                 "values.b": (count, 0), "values.c": (overlap, 0)})
    start = add("PrimitiveFloat", "Start time in seconds (all videos)", (550, 750), (320, 110), {"value": 0.})
    fps = add("PrimitiveFloat", "H3 frame rate — keep at 24", (550, 950), (320, 110), {"value": 24.})
    source_path = add("PrimitiveStringMultiline", "2. Source video path", (1280, 100), (460, 140),
                      {"value": source_video})
    mask_path = add("PrimitiveStringMultiline", "Mask video path — white regenerates", (1280, 1850), (460, 140),
                    {"value": mask_video_path})
    control_path = add("PrimitiveStringMultiline", "Motion/reference video path", (1280, 1200), (460, 140),
                       {"value": control_video})
    load_values = {"force_rate": (fps, 0), "custom_width": (width, 0), "custom_height": (height, 0),
                   "frame_load_cap": (total, 1), "start_time": (start, 0), "format": "None"}
    source = add("VHS_LoadVideoFFmpegPath", "Load source video + audio", (1840, 100), (430, 470),
                 {**load_values, "video": (source_path, 0)})
    mask_video = add("VHS_LoadVideoFFmpegPath", "Load aligned mask video", (1840, 1850), (430, 470),
                     {**load_values, "video": (mask_path, 0)})
    control = add("VHS_LoadVideoFFmpegPath", "Load motion/reference video", (1840, 1200), (430, 470),
                  {**load_values, "video": (control_path, 0)})
    mask = add("ImageToMask", "Read mask red channel", (2380, 1850), (420, 130),
               {"image": (mask_video, 0), "channel": "red"})
    threshold = add("ThresholdMask", "Binary inpaint mask", (2920, 1850), (420, 130),
                    {"mask": (mask, 0), "value": .5})
    seg_model = add("Sapiens2Loader", "Optional hair segmentation model", (3550, 1250), (420, 140),
                    {"checkpoint": "sapiens2_5b_seg.safetensors"})
    seg = add("Sapiens2Seg", "Segment clear source video", (4080, 1250), (380, 200),
              {"image": (source, 0), "sapiens2_model": (seg_model, 0), "frames_per_batch": 1})
    hair = add("Sapiens2SegExtract", "Hair → blur region mask", (4560, 1250), (470, 220),
               {"class_id_mask": (seg, 0), "class_name": "Hair", "invert": False, "num_classes": 29})
    insightface = add("H3InsightFaceLoader", "InsightFace model folder", (550, 1700), (650, 170),
                      {"model_folder": "models/insightface/buffalo_l"})
    blur = add("FaceAnonymizeVideo", "Blur source faces → Control 1", (40, 1390), (440, 1050),
               {"images": (source, 0), "insightface": (insightface, 0), "enabled": True, "gender": "any", "keep_mouth": True,
                "mouth_line": .73, "expand": .45, "mode": "blur", "strength": .5,
                "hold": 24, "ema": .6, "all_faces": True, "measure_residual": False,
                "provider": "cuda", "unload_after": True, "output_mask": False, "hair_expand": 0.,
                "region_mask": (hair, 0), "region_mode": "blur"})
    noisy = add("ImageAddNoise", "Source control noise — 0.10", (550, 1390), (300, 190),
                {"image": (blur, 0), "seed": 506229209449357, "strength": .10})
    composite = add("ImageCompositeMasked", "Composite noise into Control 1", (900, 1390), (300, 220),
                    {"destination": (blur, 0), "source": (noisy, 0), "mask": (threshold, 0),
                     "x": 0, "y": 0, "resize_source": False})
    ref = add("LoadImage", "3. Reference image — Picture 1", (2380, 100), (440, 440),
              {"image": "hybrid_windows/ref2va_stock_portrait.jpg"})
    refs = add("MMH3ImageList", "Reference pictures (add images here)", (2380, 630), (440, 170),
               {"images.image_0": (ref, 0)})
    prompts = add("PrimitiveStringMultiline", "4. Window prompts — separate with |", (2920, 100), (520, 400),
                  {"value": "The person in <Picture 1> follows the movement and timing in <Video 1>. Preserve the source video's composition, background and lighting. Natural motion, consistent appearance. | Continue the same uninterrupted shot of the person in <Picture 1>, following the movement and timing in <Video 1>. Keep the established appearance, composition, background and lighting consistent."})
    cond = add("MMH3ReferenceMultiPrompt", "Encode references + window prompts", (2920, 610), (520, 720),
               {"clip": (clip, 0), "vae": (vae, 0), "audio_vae": (avae, 0), "width": (width, 0),
                "height": (height, 0), "length": (total, 1), "ref_image_size": "match",
                "prompts": (prompts, 0), "ref_images": (refs, 0), "ref_videos.ref_video_0": (composite, 0), "ref_videos.ref_video_1": (control, 0),
                "use_input_audio": False, "unload_text_encoder": True, "window_ref_video": True,
                "chunk_frames": (window, 0), "overlap_frames": (overlap, 0)})
    refmods = add("MiniMaxH3RefModsLoader", "Optional RefMods — select saved mods", (3550, 1710), (650, 700))
    mod_cond = add("H3RefModCondSetApply", "Apply RefMods to every chunk", (4280, 1710), (730, 250),
                   {"cond_set": (cond, 0), "mods": (refmods, 0), "retention": 1.,
                    "insert_position": "before_controls", "controls_override": -1})
    add("PreviewAny", "RefMod prompt hint — describe it in your prompt", (4280, 2050), (730, 300),
        {"source": (refmods, 1)})
    encode = add("MMH3StreamingEncode", "Encode source video", (3550, 100), (420, 170),
                 {"images": (source, 0), "vae": (vae, 0), "frames_per_chunk": 85, "offload_latents": True})
    encode_audio = add("VAEEncodeAudio", "Encode source audio", (3550, 350), (420, 120),
                       {"audio": (source, 2), "vae": (avae, 0)})
    packed = add("MMH3PackAV", "Source video + audio latent", (3550, 560), (420, 170),
                 {"video_latent": (encode, 0), "audio_latent": (encode_audio, 0)})
    audio_mask = add("SolidMask", "Audio: 0 preserves / 1 regenerates", (3550, 820), (420, 170),
                     {"value": 0., "width": 1, "height": 1})
    noise = add("RandomNoise", "Seed", (4080, 100), (380, 140), {"noise_seed": 475773496218965})
    euler = add("KSamplerSelect", "Plain Euler", (4080, 310), (380, 110), {"sampler_name": "euler"})
    sigmas = add("BasicScheduler", "8-step Turbo schedule", (4080, 500), (380, 210),
                 {"model": (shift, 0), "scheduler": "simple", "steps": 8, "denoise": 1.})
    sampled = add("MMH3HybridWindowSampler", "5. Hybrid — 6 sequential + 2 joint", (4560, 100), (470, 610),
                  {"model": (shift, 0), "noise": (noise, 0), "sampler": (euler, 0), "sigmas": (sigmas, 0),
                   "cond_set": (mod_cond, 0), "latent": (packed, 0), "window_frames": (window, 0),
                   "overlap_frames": (overlap, 0), "sequential_steps": 6, "accumulator_device": "gpu",
                   "denoise_mask_mode": "max", "denoise_mask": (threshold, 0),
                   "audio_denoise_mask": (audio_mask, 0), "accepted_prefix_frames": 0, "start_window": 0})
    decoded = add("VAEDecode", "Decode video", (5140, 100), (420, 110),
                  {"samples": (sampled, 0), "vae": (vae, 0)})
    decoded_audio = add("VAEDecodeAudio", "Decode generated/preserved audio latent", (5140, 310), (420, 110),
                        {"samples": (sampled, 0), "vae": (avae, 0)})
    color = add("VideoColorStabilize", "Optional color stabilization", (5140, 540), (420, 430),
                {"images": (decoded, 0), "enabled": True, "strength": .85, "fps": (fps, 0),
                 "reference_start_seconds": 1., "reference_end_seconds": 4., "smoothing_seconds": 2.,
                 "max_tint_shift": .06})
    audio_out = add("ComfySwitchNode", "Original soundtrack? True = exact source audio", (5680, 100), (470, 180),
                    {"switch": True, "on_true": (source, 2), "on_false": (decoded_audio, 0)})
    video = add("CreateVideo", "6. Video + soundtrack", (5680, 370), (470, 240),
                {"images": (color, 0), "audio": (audio_out, 0), "fps": (fps, 0), "bit_depth": 8, "color_space": "sRGB"})
    add("SaveVideo", "Save accessible hybrid", (5680, 730), (470, 560),
        {"video": (video, 0), "filename_prefix": "video/HybridAccessible", "format": "mp4", "format.codec": "h264",
         "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 16., "codec": "auto"})
    add("PreviewAny", "Sampler report", (4560, 840), (470, 260), {"source": (sampled, 1)})
    for node_id in (mask_path, mask_video, mask, threshold):
        graph_nodes[node_id-1]["mode"] = 0 if enable_mask else 2
    for node_id in (control_path, control):
        graph_nodes[node_id-1]["mode"] = 0 if enable_control2 else 2
    for node_id in (seg_model, seg, hair):
        graph_nodes[node_id-1]["mode"] = 0 if enable_hair else 2
    # The UI keeps optional connections; the default API omits muted branches.
    muted = {str(n["id"]) for n in graph_nodes if n["mode"] == 2}
    api = {key: {**node, "inputs": {name: value for name, value in node["inputs"].items()
                                  if not (isinstance(value, list) and value[0] in muted)}}
           for key, node in api.items() if key not in muted}
    panels = (ROOT / "docs/accessible-workflow.md").read_text().split("\n<!-- workflow-panel -->\n")
    for i, panel in enumerate(panels):
        graph_nodes.append({"id": len(graph_nodes)+1, "type": "MarkdownNote",
                            "title": panel.splitlines()[0].removeprefix("# "),
                            "pos": [40 + 1560*(i % 4), 2600 + 2300*(i // 4)], "size": [1480, 2200], "flags": {},
                            "order": len(graph_nodes), "mode": 0, "inputs": [], "outputs": [],
                            "properties": {}, "widgets_values": [panel.strip()]})
    graph = {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, "HybridWindows/05-accessible-fresh-inpaint")),
             "version": .4, "revision": 0, "last_node_id": len(graph_nodes), "last_link_id": len(links),
             "nodes": graph_nodes, "links": links, "groups": [], "config": {},
             "extra": {"ds": {"scale": .3, "offset": [40, 50]}}}
    groups = [
        ("Models", [0, 30, 510, 1230]),
        ("Size and duration", [520, 30, 690, 1100]),
        ("Source video path", [1250, 30, 520, 230]),
        ("Source video + audio", [1800, 30, 510, 590]),
        ("Source blur + 0.10 noise — Control 1", [0, 1320, 1230, 1170]),
        ("OPTIONAL Hair region mask — enable group to use", [3510, 1180, 1560, 370]),
        ("OPTIONAL Control 2 — enable group to use", [1250, 1130, 1060, 600]),
        ("OPTIONAL Inpaint mask — enable group to use", [1250, 1780, 2240, 650]),
        ("OPTIONAL RefMods — leave slots at (none) to skip", [3510, 1640, 1560, 810]),
        ("Reference pictures", [2340, 30, 520, 830]),
        ("Prompts and conditioning", [2880, 30, 600, 1350]),
        ("Clear source encode and audio mask", [3510, 30, 510, 1030]),
        ("Hybrid sampling", [4040, 30, 1030, 1120]),
        ("Decode and color", [5100, 30, 500, 1010]),
        ("Soundtrack and output", [5640, 30, 550, 1310]),
        ("Instructions — setup, chunks, blur and masks", [0, 2530, 6240, 4620]),
    ]
    graph["groups"] = [{"id": i+1, "title": name, "bounding": bounds,
                        "color": "#3f789e", "font_size": 24}
                       for i, (name, bounds) in enumerate(groups)]
    return graph, api


async def main():
    local_file = COMFY / "user/default/workflows/H3 Diagnostics" / f"{NAME}.json"
    existing = json.loads(local_file.read_text()) if local_file.exists() else None
    graph, api = await build()
    folder = ROOT / "example_workflows"
    (folder / f"{NAME}.json").write_text(json.dumps(graph, indent=2) + "\n")
    (folder / f"{NAME}.api.json").write_text(json.dumps(api, indent=2) + "\n")
    # Prefill only the local copy from the existing 05; the shareable copy has no personal paths.
    if SOURCE.exists():
        original = json.loads(SOURCE.read_text())
        saved = {n["id"]: n for n in original["nodes"]}
        titles = {"2. Source video path": 257, "Mask video path — white regenerates": 443,
                  "Motion/reference video path": 255}
        for n in graph["nodes"]:
            if n.get("title") in titles:
                value = saved[titles[n["title"]]]["widgets_values_named"]["path"]
                n["widgets_values"] = [value]
                n["widgets_values_named"] = {"value": value}
    if existing:
        old = {(n.get("title"), n["type"]): n for n in existing["nodes"]}
        for n in graph["nodes"]:
            prior = old.get((n.get("title"), n["type"]))
            if prior is None or n["type"] == "MarkdownNote":
                continue
            # Preserve edited widgets, never old input links or node modes.
            if "widgets_values_named" in prior:
                for socket in n.get("inputs", []):
                    name = socket["name"]
                    if socket.get("link") is None and name in prior["widgets_values_named"]:
                        n["widgets_values_named"][name] = prior["widgets_values_named"][name]
                if isinstance(n["widgets_values"], dict):
                    n["widgets_values"].update(n["widgets_values_named"])
                elif prior.get("widgets_values_named", {}).keys() == n.get("widgets_values_named", {}).keys():
                    n["widgets_values"] = prior["widgets_values"]
    destination = COMFY / "user/default/workflows/H3 Diagnostics"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / f"{NAME}.json").write_text(json.dumps(graph, indent=2) + "\n")
    print(f"Built {NAME}: {len(api)} processing nodes.")


if __name__ == "__main__":
    asyncio.run(main())
