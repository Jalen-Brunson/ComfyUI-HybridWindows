"""Validate the saved example with ComfyUI, without loading model weights."""

import asyncio
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("hybrid_workflow_builder", ROOT / "build_workflow.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
import execution


def intersects(a, b):
    return a[0] < b[0]+b[2] and b[0] < a[0]+a[2] and a[1] < b[1]+b[3] and b[1] < a[1]+a[3]


async def validate(mode):
    graph, api = await builder.build(mode)
    result = await execution.validate_prompt("hybrid-native-validation", api, None)
    assert result[0], json.dumps(result, indent=2, default=str)
    by_id = {n["id"]: n for n in graph["nodes"]}
    for link_id, source, source_slot, target, target_slot, typ in graph["links"]:
        assert by_id[target]["inputs"][target_slot]["link"] == link_id
        assert link_id in by_id[source]["outputs"][source_slot]["links"]
        assert by_id[source]["outputs"][source_slot]["type"] == typ
        assert by_id[target]["inputs"][target_slot]["type"] == typ
    rectangles = [(n["id"], [n["pos"][0], n["pos"][1]-30, n["size"][0], n["size"][1]+30]) for n in graph["nodes"]]
    for i, (node_id, rect) in enumerate(rectangles):
        for other_id, other_rect in rectangles[i+1:]:
            assert not intersects(rect, other_rect), f"Overlapping nodes: {node_id}, {other_id}"
    for i, group in enumerate(graph["groups"]):
        for other in graph["groups"][i+1:]:
            assert not intersects(group["bounding"], other["bounding"]), (group["title"], other["title"])
    for node_id, rect in rectangles:
        assert any(rect[0] >= g["bounding"][0] and rect[1] >= g["bounding"][1]
                   and rect[0]+rect[2] <= g["bounding"][0]+g["bounding"][2]
                   and rect[1]+rect[3] <= g["bounding"][1]+g["bounding"][3]
                   for g in graph["groups"]), f"Node {node_id} escapes its group"
    types = [n["type"] for n in graph["nodes"]]
    assert types.count("MarkdownNote") == 1
    assert types.count("KSamplerAdvanced") == 2
    assert types.count("PrimitiveStringMultiline") == 3
    loaders = [n for n in graph["nodes"] if n["type"] == "LoadImage"]
    for loader in loaders:
        filename = loader["widgets_values_named"]["image"]
        assert filename.startswith("hybrid_windows/"), filename
        assert (ROOT / "example_workflows" / "assets" / Path(filename).name).is_file(), filename
    if mode == "fl2va":
        assert len(loaders) == 2
        conds = [n for n in api.values() if n["class_type"] == "MiniMaxH3ImageToVideo"]
        assert len(conds) == 3
        assert conds[0]["inputs"]["first_frame"] == [str(loaders[0]["id"]), 0]
        assert not {"first_frame", "last_frame"} & conds[1]["inputs"].keys()
        assert conds[2]["inputs"]["last_frame"] == [str(loaders[1]["id"]), 0]
        assert "last_frame" not in conds[0]["inputs"] and "first_frame" not in conds[2]["inputs"]
    else:
        assert len(loaders) == 1
        conds = [n for n in api.values() if n["class_type"] == "MiniMaxH3ReferenceToVideo"]
        assert len(conds) == 3
        assert all(c["inputs"]["ref_images.ref_image_0"] == [str(loaders[0]["id"]), 0] for c in conds)
    custom = {"H3HybridWindows", "VideoColorStabilize"}
    color_id = next(k for k, n in api.items() if n["class_type"] == "VideoColorStabilize")
    color_inputs = api[color_id]["inputs"]
    create = next(n["inputs"] for n in api.values() if n["class_type"] == "CreateVideo")
    assert create["images"] == [color_id, 0]
    assert create["fps"] == color_inputs["fps"]
    assert api[color_inputs["images"][0]]["class_type"] == "VAEDecode"
    assert api[create["audio"][0]]["class_type"] == "VAEDecodeAudio"
    assert "source_frames" not in color_inputs
    assert not {"H3HybridControlNet", "ModelPatchLoader", "ImageAddNoise", "LoadVideo"} & set(types)
    frontend = {"MarkdownNote"}
    assert set(types)-frontend-set(builder.nodes.NODE_CLASS_MAPPINGS) == set()
    assert not any(n["class_type"] in frontend for n in api.values())
    for name in set(types)-custom-frontend:
        cls = builder.nodes.NODE_CLASS_MAPPINGS[name]
        assert cls.__module__ == "nodes" or cls.__module__.startswith("comfy_extras."), (name, cls.__module__)
    report = {"mode": mode, "comfy_prompt_valid": True, "nodes": len(types), "native_nodes": len(api)-len(custom),
              "frontend_notes": types.count("MarkdownNote"),
              "custom_node_types": sorted(custom), "links": len(graph["links"]),
              "node_and_group_overlap": False, "gpu_render": "not run"}
    print(json.dumps(report, indent=2))


async def main():
    for mode in ("ref2va", "fl2va"):
        await validate(mode)


if __name__ == "__main__":
    asyncio.run(main())
