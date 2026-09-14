"""Build the portable V2V workflow against installed ComfyUI node schemas."""

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

NAME = "V2V Hybrid Sampling with inpainting option"



async def build(source_video="input/source.mp4", mask_video_path="input/mask.webm", control_video="input/control.mp4", *, enable_mask=False, enable_control2=False, enable_blur=True, reference_mode=0):
    server.PromptServer(asyncio.get_running_loop())
    nodes.NODE_CLASS_MAPPINGS.update(nodes_preview_any.NODE_CLASS_MAPPINGS)
    for module in (nodes_audio, nodes_minimax_h3, nodes_video, nodes_custom_sampler,
                   nodes_mask, nodes_primitive, nodes_math, nodes_logic, nodes_images):
        extension = await module.comfy_entrypoint()
        for cls in await extension.get_node_list():
            nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls
    for folder in ("ComfyUI-MMH3Tools", "comfyui-videohelpersuite", "ComfyUI-HybridWindows", "h3_face_tools", "ComfyUI-Sapiens2", "ComfyUI-MiniMaxH3Mod"):
        folder_path = (str(ROOT) if folder == "ComfyUI-HybridWindows" else
                       os.environ.get("MMH3TOOLS_TEST_ROOT", str(Path(os.environ.get("HYBRID_DEPS_ROOT", str(COMFY / "custom_nodes"))) / folder)) if folder == "ComfyUI-MMH3Tools" else
                       str(Path(os.environ.get("HYBRID_DEPS_ROOT", str(COMFY / "custom_nodes"))) / folder))
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
        elif kind.startswith("H3Hybrid") or kind in ("H3SegmentedVideoBlur", "VideoColorStabilize"):
            node["properties"]["aux_id"] = "Jalen-Brunson/ComfyUI-HybridWindows"
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

    source_path = add('PrimitiveStringMultiline', 'Source video path', (40, 100), (960, 160), {'value': source_video})
    project = add('PrimitiveString', 'Project name — keep this name when continuing', (40, 320), (960, 100), {'value': 'my_v2v_project'})
    prompts = add('PrimitiveStringMultiline', 'Prompt — one for all chunks, or separate global chunk prompts with |', (40, 490), (960, 330),
                  {'value': 'The person in <Picture 1> follows the movement and timing in <Video 1>. Preserve the composition, background, lighting and lip movement of the source video. Natural movement and consistent appearance throughout one uninterrupted shot.'})
    plan = add('H3HybridRunPlan', 'Run controls — fresh, continue, or redo later chunks', (40, 900), (960, 480),
               {'source_video': (source_path, 0), 'prompts': (prompts, 0), 'accepted_chunks': 0, 'new_chunks': 2,
                'chunk_frames': 243, 'overlap_frames': 39, 'source_start_seconds': 0., 'max_frames': 0})
    width = add('PrimitiveInt', 'Output width', (40, 1460), (440, 110), {'value': 832})
    height = add('PrimitiveInt', 'Output height', (560, 1460), (440, 110), {'value': 480})
    inpaint_on = add('PrimitiveBoolean', 'Enable inpainting', (40, 1650), (440, 110), {'value': enable_mask})
    blur_on = add('PrimitiveBoolean', 'Enable segmentation blur', (560, 1650), (440, 110), {'value': enable_blur})
    window = add('PrimitiveInt', 'Window frames — link also feeds Run controls', (40, 1840), (440, 110), {'value': 243})
    overlap = add('PrimitiveInt', 'Overlap frames', (560, 1840), (440, 110), {'value': 39})
    # Link controls created later without duplicating their values in the run planner.
    def wire(origin, output, target, name):
        a, b = graph_nodes[origin-1], graph_nodes[target-1]
        slot = next(i for i,s in enumerate(b['inputs']) if s['name'] == name)
        lid = len(links) + 1
        links.append([lid, origin, output, target, slot, a['outputs'][output]['type']])
        a['outputs'][output]['links'].append(lid)
        b['inputs'][slot]['link'] = lid
        api[str(target)]['inputs'][name] = [str(origin), output]
    wire(window, 0, plan, 'chunk_frames'); wire(overlap, 0, plan, 'overlap_frames')
    add('PreviewAny', 'Run plan and next accepted-chunk count', (40, 2050), (960, 270), {'source': (plan, 8)})

    base = add('UNETLoader', 'H3 Ref2VA model', (1140, 100), (960, 140),
               {'unet_name': 'minimax_h3_ref2va_int8_convrot.safetensors', 'weight_dtype': 'default'})
    lora = add('LoraLoaderModelOnly', 'Native PDD eight-step LoRA', (1140, 310), (960, 150),
               {'model': (base, 0), 'lora_name': 'MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors', 'strength_model': 1.})
    shift = add('MiniMaxH3SigmaShift', 'Video / audio sigma shifts', (1140, 530), (960, 150),
                {'model': (lora, 0), 'shift_video': 12., 'shift_audio': 3.})
    clip = add('CLIPLoader', 'H3 text encoder', (1140, 750), (960, 150),
               {'clip_name': 'qwen3vl_32b_minimax_h3_int8_convrot.safetensors', 'type': 'minimax', 'device': 'default'})
    vae = add('VAELoader', 'Video VAE', (1140, 970), (960, 110), {'vae_name': 'minimax_h3_video_vae_int8_convrot.safetensors'})
    avae = add('VAELoader', 'Audio VAE', (1140, 1150), (960, 110), {'vae_name': 'minimax_h3_audio_vae_fp32.safetensors'})
    seed = add('PrimitiveInt', 'Seed — keep fixed when comparing or continuing', (1140, 1350), (960, 110), {'value': 123})
    steps = add('PrimitiveInt', 'Total sampling steps', (1140, 1550), (440, 110), {'value': 8})
    split = add('PrimitiveInt', 'Switch to joint sampling at step', (1660, 1550), (440, 110), {'value': 6})

    load_values = {'force_rate': 24., 'custom_width': (width, 0), 'custom_height': (height, 0),
                   'frame_load_cap': (plan, 2), 'start_time': (plan, 1), 'format': 'None'}
    source = add('VHS_LoadVideoFFmpegPath', 'Load this run from the clear source + audio', (2240, 100), (960, 470),
                 {**load_values, 'video': (source_path, 0)})
    mask_path = add('PrimitiveStringMultiline', 'Inpaint mask path — used only when enabled', (1140, 1830), (960, 160), {'value': mask_video_path})
    mask_video = add('VHS_LoadVideoFFmpegPath', 'Inpaint mask video — only loaded when enabled', (2240, 670), (960, 470),
                     {**load_values, 'video': (mask_path, 0)})
    mask = add('ImageToMask', 'Read mask red channel', (2240, 1220), (440, 120), {'image': (mask_video, 0), 'channel': 'red'})
    threshold = add('ThresholdMask', 'White regenerates / black preserves', (2760, 1220), (440, 120), {'mask': (mask, 0), 'value': .5})
    control = add('VHS_LoadVideoFFmpegPath', 'Optional second aligned control video', (2240, 1450), (960, 470),
                  {**load_values, 'video': control_video})
    graph_nodes[control-1]['mode'] = 0 if enable_control2 else 2
    encode = add('MMH3StreamingEncode', 'Encode the clear source', (2240, 2040), (960, 170),
                 {'images': (source, 0), 'vae': (vae, 0), 'frames_per_chunk': 85, 'offload_latents': True})
    encode_audio = add('VAEEncodeAudio', 'Encode source audio', (2240, 2310), (960, 110), {'audio': (source, 2), 'vae': (avae, 0)})
    packed = add('MMH3PackAV', 'Pack source video and audio', (2240, 2520), (960, 170),
                 {'video_latent': (encode, 0), 'audio_latent': (encode_audio, 0)})

    prior = add('H3HybridResumeLoad', 'Select continuation — latest completed run for this project', (5540, 100), (960, 210),
                {'run': (plan, 0), 'project': (project, 0)})
    prefix = add('VHS_LoadVideoFFmpegPath', 'Load accepted prefix — skipped on fresh runs', (5540, 420), (960, 470),
                 {'video': (prior, 1), 'force_rate': 24., 'custom_width': (width, 0), 'custom_height': (height, 0),
                  'frame_load_cap': (plan, 6), 'start_time': 0., 'format': 'None'})
    prepared = add('H3HybridResumeLatent', 'Prepare source preservation and the pinned continuation window', (2240, 2790), (960, 230),
                   {'source': (packed, 0), 'run': (plan, 0), 'prior': (prior, 0), 'inpainting': (inpaint_on, 0), 'mask': (threshold, 0)})

    seg_model = add('Sapiens2Loader', 'Sapiens2 segmentation model', (40, 3460), (600, 140), {'checkpoint': 'sapiens2_5b_seg.safetensors'})
    seg = add('Sapiens2Seg', 'Segment clear source — skipped when blur is off', (700, 3460), (600, 180),
              {'image': (source, 0), 'sapiens2_model': (seg_model, 0), 'frames_per_batch': 1})
    seg_kw = {'class_id_mask': (seg, 0), 'invert': False, 'num_classes': 29}
    classes = {}
    for i, name in enumerate(('Face_Neck', 'Hair', 'Eyeglass', 'Upper_Lip', 'Lower_Lip', 'Upper_Teeth', 'Lower_Teeth', 'Tongue')):
        classes[name] = add('Sapiens2SegExtract', name, (40 + (i % 4) * 660, 3780 + (i // 4) * 260), (600, 180), {**seg_kw, 'class_name': name})
    def union(a, b, title, x, y):
        return add('MaskComposite', title, (x, y), (600, 180), {'destination': (a, 0), 'source': (b, 0), 'x': 0, 'y': 0, 'operation': 'add'})
    face_hair = union(classes['Face_Neck'], classes['Hair'], 'Face + hair', 40, 4360)
    region = union(face_hair, classes['Eyeglass'], 'Face + hair + glasses → blur', 700, 4360)
    lips = union(classes['Upper_Lip'], classes['Lower_Lip'], 'Lips → protect', 1360, 4360)
    teeth = union(classes['Upper_Teeth'], classes['Lower_Teeth'], 'Teeth → protect', 2020, 4360)
    mouth = union(lips, teeth, 'Lips + teeth → protect', 1360, 4620)
    protection = union(mouth, classes['Tongue'], 'Mouth protection', 2020, 4620)
    blur = add('H3SegmentedVideoBlur', 'Segmentation blur — preserve lips, hands and props', (2840, 3460), (900, 330),
               {'images': (source, 0), 'enabled': (blur_on, 0), 'region_mask': (region, 0), 'protect_mask': (protection, 0),
                'strength': .5, 'region_grow': 2, 'protect_grow': 2, 'min_component': .05})
    noisy = add('ImageAddNoise', 'Control image noise', (2840, 3910), (900, 180),
                {'image': (blur, 0), 'seed': (seed, 0), 'strength': .10})
    composite = add('ImageCompositeMasked', 'Noise follows the inpaint region when enabled', (2840, 4210), (900, 210),
                    {'destination': (blur, 0), 'source': (noisy, 0), 'mask': (prepared, 1), 'x': 0, 'y': 0, 'resize_source': False})

    ref = add('LoadImage', 'Reference image — Picture 1 (optional; mute or bypass for RefMod only)', (3340, 100), (960, 630), {'image': 'hybrid_windows/ref2va_stock_portrait.jpg'})
    assert reference_mode in (0, 2, 4)  # Always, Mute, Bypass; LoadImage has no passthrough image.
    graph_nodes[ref - 1]['mode'] = reference_mode
    cond = add('MMH3ReferenceMultiPrompt', 'Encode the planned prompts and aligned controls', (3340, 900), (960, 720),
               {'clip': (clip, 0), 'vae': (vae, 0), 'audio_vae': (avae, 0), 'width': (width, 0), 'height': (height, 0),
                'length': (plan, 2), 'ref_image_size': 'match', 'prompts': (plan, 7), 'ref_images': (ref, 0),
                'ref_videos.ref_video_0': (composite, 0), 'ref_videos.ref_video_1': (control, 0),
                'use_input_audio': False, 'unload_text_encoder': True, 'window_ref_video': True,
                'chunk_frames': (window, 0), 'overlap_frames': (overlap, 0)})
    refmods = add('MiniMaxH3RefModsLoader', 'Optional RefMods — leave unused slots at (none)', (3340, 1730), (960, 700))
    mod_cond = add('H3RefModCondSetApply', 'Apply selected RefMods to every window', (3340, 2540), (960, 250),
                   {'cond_set': (cond, 0), 'mods': (refmods, 0), 'retention': 1., 'insert_position': 'before_controls', 'controls_override': -1})
    add('PreviewAny', 'RefMod prompt hint', (3340, 2890), (960, 240), {'source': (refmods, 1)})

    audio_mask = add('SolidMask', 'Audio mask — 0 keeps source / 1 regenerates', (4440, 100), (960, 170), {'value': 0., 'width': 1, 'height': 1})
    hybrid = add('H3HybridWindows', 'Hybrid sampling — context 5, automatic inpaint pinning', (4440, 380), (960, 680),
                 {'model': (shift, 0), 'window_frames': (window, 0), 'overlap_frames': (overlap, 0),
                  'total_steps': (steps, 0), 'cond_set': (mod_cond, 0), 'latent': (prepared, 0),
                  'denoise_mask': (prepared, 1), 'audio_denoise_mask': (audio_mask, 0), 'denoise_mask_mode': 'max',
                  'accepted_prefix_frames': (plan, 5), 'start_window': (plan, 4), 'noise_mode': 'per_window',
                  'overlap_pin': (prepared, 2), 'context_frames': 5})
    negative = add('ConditioningZeroOut', 'Empty negative conditioning', (4440, 1160), (960, 100), {'conditioning': (hybrid, 2)})
    common = {'noise_seed': (seed, 0), 'steps': (steps, 0), 'cfg': 1., 'sampler_name': 'euler', 'scheduler': 'simple',
              'positive': (hybrid, 2), 'negative': (negative, 0)}
    sequential = add('KSamplerAdvanced', 'Sequential stage', (4440, 1370), (960, 480),
                     {**common, 'model': (hybrid, 0), 'latent_image': (hybrid, 4), 'add_noise': 'enable',
                      'start_at_step': 0, 'end_at_step': (split, 0), 'return_with_leftover_noise': 'enable'})
    sampled = add('KSamplerAdvanced', 'Joint finishing stage', (4440, 1960), (960, 480),
                  {**common, 'model': (hybrid, 1), 'latent_image': (sequential, 0), 'add_noise': 'disable',
                   'start_at_step': (split, 0), 'end_at_step': (steps, 0), 'return_with_leftover_noise': 'disable'})
    add('PreviewAny', 'Hybrid sampler report', (4440, 2540), (960, 240), {'source': (hybrid, 5)})

    decoded = add('VAEDecode', 'Decode this run', (5540, 1000), (440, 110), {'samples': (sampled, 0), 'vae': (vae, 0)})
    decoded_audio = add('VAEDecodeAudio', 'Decode generated audio', (6060, 1000), (440, 110), {'samples': (sampled, 0), 'vae': (avae, 0)})
    audio_out = add('ComfySwitchNode', 'Use source soundtrack? True keeps the original waveform', (5540, 1220), (960, 180),
                    {'switch': True, 'on_true': (source, 2), 'on_false': (decoded_audio, 0)})
    assembled = add('H3HybridResumeOutput', 'Preserve accepted pixels/audio and append new frames', (5540, 1510), (960, 210),
                    {'images': (decoded, 0), 'audio': (audio_out, 0), 'run': (plan, 0), 'prefix_images': (prefix, 0), 'prefix_audio': (prefix, 2)})
    video = add('CreateVideo', 'Complete video', (5540, 1830), (960, 220),
                {'images': (assembled, 0), 'audio': (assembled, 1), 'fps': 24., 'bit_depth': 8, 'color_space': 'sRGB'})
    save = add('H3HybridSaveRun', 'Save full video + matching continuation master', (5540, 2160), (960, 420),
               {'video': (video, 0), 'latent': (sampled, 0), 'prior': (prior, 0), 'run': (plan, 0), 'project': (project, 0),
                'filename_prefix': 'video/V2V_Hybrid', 'format': 'mp4', 'format.codec': 'h264',
                'format.codec.encoding': 're-encode', 'format.codec.encoding.crf': 16., 'codec': 'auto'})
    add('PreviewAny', 'Saved run and next continuation setting', (5540, 2690), (960, 260), {'source': (save, 3)})

    muted = {str(n['id']) for n in graph_nodes if n['mode'] in (2, 4)}
    api = {key: {**node, 'inputs': {name: value for name, value in node['inputs'].items()
                if not (isinstance(value, list) and value[0] in muted)}} for key, node in api.items() if key not in muted}
    panels = (ROOT / 'docs/v2v-workflow.md').read_text().split('\n<!-- workflow-panel -->\n')
    for i, panel in enumerate(panels):
        graph_nodes.append({'id': len(graph_nodes)+1, 'type': 'MarkdownNote', 'title': panel.splitlines()[0].lstrip('# '),
                            'pos': [40 + 2200*(i % 3), 5020 + 1900*(i // 3)], 'size': [2060, 1770],
                            'flags': {}, 'order': len(graph_nodes), 'mode': 0, 'inputs': [], 'outputs': [],
                            'properties': {}, 'widgets_values': [panel.strip()]})
    groups = [('1 · Source and run controls', [0, 30, 1040, 2390]),
              ('2 · Model files, settings and optional mask path', [1100, 30, 1040, 2060]),
              ('3 · Clear source and inpaint preparation', [2200, 30, 1040, 3110]),
              ('4 · Reference pictures and prompts', [3300, 30, 1040, 3200]),
              ('5 · Sequential + joint hybrid sampling', [4400, 30, 1040, 2860]),
              ('6 · Save and continue', [5500, 30, 1040, 3020]),
              ('Segmentation blur · face + hair, mouth protected', [0, 3360, 3790, 1540]),
              ('Instructions', [0, 4920, 6540, 3850])]
    graph = {'id': str(uuid.uuid5(uuid.NAMESPACE_URL, 'HybridWindows/V2V-Hybrid-Sampling-with-inpainting-option')),
             'version': .4, 'revision': 0, 'last_node_id': len(graph_nodes), 'last_link_id': len(links),
             'nodes': graph_nodes, 'links': links,
             'groups': [{'id': i+1, 'title': name, 'bounding': box, 'color': '#3f789e', 'font_size': 24} for i,(name,box) in enumerate(groups)],
             'config': {}, 'extra': {'ds': {'scale': .32, 'offset': [40, 50]}}}
    return graph, api


async def main():
    graph, api = await build()
    folder = ROOT / 'example_workflows'
    (folder / f'{NAME}.json').write_text(json.dumps(graph, indent=2) + '\n')
    (folder / f'{NAME}.api.json').write_text(json.dumps(api, indent=2) + '\n')
    print(f'Built {NAME}: {len(api)} processing nodes.')


if __name__ == '__main__':
    asyncio.run(main())
