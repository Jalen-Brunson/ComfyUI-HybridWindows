"""Check the derived 05N graph against the 05 it came from.

The point of the derivation is that only the sampler changes, so most of this
asserts what did NOT change: every surviving node byte-for-byte, the link table
on both ends, and the untouched subgraph definitions.
"""

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("derive_05n", ROOT / "derive_05n_workflow.py")
derive = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = derive
spec.loader.exec_module(derive)
import nodes

FRONTEND_ONLY = {"MarkdownNote", "Note"}


def intersects(a, b):
    return a[0] < b[0]+b[2] and b[0] < a[0]+a[2] and a[1] < b[1]+b[3] and b[1] < a[1]+a[3]


async def main():
    await derive.register()
    source = json.loads(derive.SOURCE.read_text())
    graph = derive.derive().data
    old = {n["id"]: n for n in source["nodes"]}
    new = {n["id"]: n for n in graph["nodes"]}
    added = sorted(set(new) - set(old))
    assert set(old) - set(new) == {derive.SAMPLER_ID}, "only the sampler may be removed"
    assert len(added) == 8, added

    # Everything else is the 05 file's own node, unchanged. Get_sampler is the one
    # exception: the second sampler shares its selector, so it gains a link.
    sampler_getter = source["links"] and next(
        l[1] for l in source["links"] if l[0] == old[derive.SAMPLER_ID]["inputs"][2]["link"])
    for node_id, node in old.items():
        if node_id == derive.SAMPLER_ID:
            continue
        if node_id == sampler_getter:
            assert len(new[node_id]["outputs"][0]["links"]) == len(node["outputs"][0]["links"]) + 1
            continue
        assert new[node_id] == node, f"node {node_id} ({node['type']}) changed"

    # Link table agrees with both endpoints. Set/Get and the easy switches carry
    # a wildcard socket, so a link's declared type may be narrower than theirs;
    # what matters is that this derivation adds no new mismatch.
    def loose(graph_data):
        """Links whose declared type is narrower than a socket's (wildcards,
        multi-type inputs). Pre-existing in 05; this must not add any."""
        by_id = {n["id"]: n for n in graph_data["nodes"]}
        return sorted(l[0] for l in graph_data["links"]
                      if by_id[l[1]]["outputs"][l[2]]["type"] != l[5]
                      or by_id[l[3]]["inputs"][l[4]]["type"] != l[5])
    for link_id, src, src_slot, dst, dst_slot, typ in graph["links"]:
        assert new[dst]["inputs"][dst_slot]["link"] == link_id, (link_id, dst)
        assert link_id in new[src]["outputs"][src_slot]["links"], (link_id, src)
    assert loose(graph) == loose(source), "the derivation changed a socket type"
    seen = {l[0] for l in graph["links"]}
    for node in graph["nodes"]:
        for socket in node["inputs"]:
            assert socket.get("link") in seen or socket.get("link") is None, (node["id"], socket["name"])

    by_type = {}
    for node_id in added:
        by_type.setdefault(new[node_id]["type"], []).append(node_id)
    windows_id = by_type["H3HybridWindows"][0]
    split_id = by_type["SplitSigmas"][0]
    warm_id, joint_id = sorted(by_type["SamplerCustomAdvanced"])
    windows, split = new[windows_id], new[split_id]

    def source_of(node, name):
        slot = next(i for i, s in enumerate(node["inputs"]) if s["name"] == name)
        link = next(l for l in graph["links"] if l[0] == node["inputs"][slot]["link"])
        return link[1], link[2]

    # The 05 wires reach the same values, through the new nodes.
    getter = lambda node_id: new[node_id].get("widgets_values", [None])[0]
    expected = {"model": None, "cond_set": None, "latent": "master_latent",
                "denoise_mask": "denoise_mask", "audio_denoise_mask": "audio_mask",
                "window_frames": "chunk_frames", "overlap_frames": "overlap_frames"}
    for name, wanted in expected.items():
        origin = source_of(windows, name)[0]
        if wanted is not None:
            assert new[origin]["type"] == "GetNode" and getter(origin) == wanted, (name, origin)
    assert new[source_of(split, "sigmas")[0]]["type"] == "GetNode"
    assert getter(source_of(split, "sigmas")[0]) == "sigmas"
    assert source_of(new[warm_id], "latent_image") == (windows_id, 4), "sample the node's latent output"
    assert source_of(new[joint_id], "latent_image") == (warm_id, 0), "joint must take `output`, not `denoised_output`"
    assert source_of(new[warm_id], "sigmas") == (split_id, 0)
    assert source_of(new[joint_id], "sigmas") == (split_id, 1)
    assert new[source_of(new[joint_id], "noise")[0]]["type"] == "DisableNoise"
    assert new[source_of(new[warm_id], "noise")[0]]["type"] == "RandomNoise"

    named = windows["widgets_values_named"]
    assert named["total_steps"] == derive.TOTAL_STEPS, named
    assert named["noise_mode"] == "per_window", named
    assert split["widgets_values_named"]["step"] == derive.SPLIT_STEP

    # Downstream reads the joint stage; the report comes from the node.
    consumers = {l[3] for l in graph["links"] if l[1] == joint_id}
    assert {new[c]["type"] for c in consumers} >= {"VAEDecode", "VAEDecodeAudio", "H3MergeMasters"}
    report_targets = [l[3] for l in graph["links"] if l[1] == windows_id and l[2] == 5]
    assert report_targets and all(new[t]["type"] == "PreviewAny" for t in report_targets)

    # Layout: the new nodes live inside the new group and nothing else does.
    group = next(g for g in graph["groups"] if g["bounding"] == list(derive.GROUP))
    for node_id in added:
        pos, size = new[node_id]["pos"], new[node_id]["size"]
        assert (group["bounding"][0] <= pos[0] and pos[0]+size[0] <= group["bounding"][0]+group["bounding"][2]
                and group["bounding"][1] <= pos[1]-30
                and pos[1]+size[1] <= group["bounding"][1]+group["bounding"][3]), node_id
    boxes = [(n["id"], [n["pos"][0], n["pos"][1]-30, n["size"][0], n["size"][1]+30])
             for n in graph["nodes"] if n["id"] in added]
    for index, (node_id, box) in enumerate(boxes):
        for other_id, other in boxes[index+1:]:
            assert not intersects(box, other), (node_id, other_id)
    for other in graph["groups"]:
        if other is not group:
            assert not intersects(group["bounding"], other["bounding"]), other.get("title")

    # Untouched: the subgraph definitions and the frontend's own keys.
    assert graph["definitions"] == source["definitions"]
    assert graph["extra"] == source["extra"]
    assert graph["last_node_id"] == max(new) and graph["last_link_id"] == max(seen)

    registered = [n["type"] for n in graph["nodes"]
                  if n["type"] not in FRONTEND_ONLY and "-" not in n["type"]
                  and n["type"] not in ("GetNode", "SetNode")]
    missing = sorted({t for t in registered if t not in nodes.NODE_CLASS_MAPPINGS})
    valid, error = await derive.api_schema_check()
    assert valid is not False, error
    print(json.dumps({
        "nodes": len(graph["nodes"]), "links": len(graph["links"]),
        "added": added, "unchanged_nodes": len(old) - 1,
        "cond_set_only_prompt_validates": valid if valid is not None else error,
        "unregistered_types_needing_a_running_server": missing,
        "node_and_group_overlap": False}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
