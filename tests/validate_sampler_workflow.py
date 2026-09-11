"""Validate the single-sampler graph without loading model weights."""

import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_sampler_workflow import build, nodes
import execution


async def main():
    graph, api = await build()
    result = await execution.validate_prompt("hybrid-sampler-validation", api, None)
    assert result[0], json.dumps(result, indent=2, default=str)
    by_id = {n["id"]: n for n in graph["nodes"]}
    for lid, source, output, target, slot, kind in graph["links"]:
        assert by_id[target]["inputs"][slot]["link"] == lid
        assert lid in by_id[source]["outputs"][output]["links"]
        assert by_id[source]["outputs"][output]["type"] == kind
        assert by_id[target]["inputs"][slot]["type"] == kind
    for i, a in enumerate(graph["nodes"]):
        x, y = a["pos"]; w, h = a["size"]
        for b in graph["nodes"][i+1:]:
            bx, by = b["pos"]; bw, bh = b["size"]
            assert not (x < bx+bw and bx < x+w and y-30 < by+bh and by-30 < y+h), (a["id"], b["id"])
    custom = {"MMH3HybridWindowSampler", "MMH3CondToSet"}
    for n in api.values():
        if n["class_type"] not in custom:
            module = nodes.NODE_CLASS_MAPPINGS[n["class_type"]].__module__
            assert module == "nodes" or module.startswith("comfy_extras."), module
    by_type = {n["class_type"]: n["inputs"] for n in api.values()}
    sampler = by_type["MMH3HybridWindowSampler"]
    count = by_type["MMH3CondToSet"]["count"]
    assert by_type["EmptyMiniMaxH3LatentAV"]["length"] == sampler["window_frames"] + (count-1)*(sampler["window_frames"]-sampler["overlap_frames"])
    ref = by_type["MiniMaxH3ReferenceToVideo"]["ref_images.ref_image_0"]
    assert api[ref[0]]["class_type"] == "LoadImage"
    assert (ROOT / "example_workflows/assets" / Path(by_type["LoadImage"]["image"]).name).is_file()
    assert api[by_type["CreateVideo"]["images"][0]]["class_type"] == "VAEDecode"
    assert api[by_type["CreateVideo"]["audio"][0]]["class_type"] == "VAEDecodeAudio"
    print(f"Valid: {len(api)} processing nodes, {len(api)-len(custom)} native, {len(graph['links'])} links; no node overlaps.")


if __name__ == "__main__":
    asyncio.run(main())
