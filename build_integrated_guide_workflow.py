"""Replace external per-stage video guide adapters in a saved V2V workflow."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import uuid


def convert(workflow):
    graph = copy.deepcopy(workflow)
    nodes = {node["id"]: node for node in graph["nodes"]}
    links = {link[0]: link for link in graph["links"]}
    adapters = [node for node in graph["nodes"] if node["type"] == "H3CharacterSwapGuides"]
    if not adapters:
        raise ValueError("No H3CharacterSwapGuides adapters in this workflow.")

    def source(node, name):
        port = next((port for port in node["inputs"] if port["name"] == name), {})
        link = links.get(port.get("link"))
        return tuple(link[1:3]) if link else None

    positive = source(adapters[0], "positive")
    if positive is None or nodes[positive[0]]["type"] != "H3HybridWindows":
        raise ValueError("Guide adapters must consume H3 Hybrid Windows conditioning.")
    hybrid = nodes[positive[0]]
    for adapter in adapters:
        for name in ("positive", "source_frames", "vae", "source_audio", "audio_vae", "swap_config"):
            if source(adapter, name) != source(adapters[0], name):
                raise ValueError(f"Guide adapters use different {name} inputs; they cannot share one guide.")
    config_id = source(adapters[0], "swap_config")[0]
    config = nodes[config_id]
    subgraphs = graph.get("definitions", {}).get("subgraphs", [])
    config_graph = next((item for item in subgraphs if item["id"] == config["type"]), None)
    settings = config if config["type"] == "H3CharacterSwapSetup" else next(
        node for node in config_graph["nodes"] if node["type"] == "H3CharacterSwapSetup")
    named = settings.get("widgets_values_named", {})
    if not named.get("enabled", settings["widgets_values"][0]):
        raise ValueError("Enable the source guide configuration before converting it.")
    use_audio = named.get("guide_audio", settings["widgets_values"][5])
    connections = [("guide_video", "IMAGE", source(adapters[0], "source_frames")),
                   ("guide_vae", "VAE", source(adapters[0], "vae")),
                   ("guide_audio", "AUDIO", source(adapters[0], "source_audio") if use_audio else None),
                   ("guide_audio_vae", "VAE", source(adapters[0], "audio_vae") if use_audio else None)]
    remove = {node["id"] for node in adapters}
    for link in graph["links"]:
        if link[1] in remove and link[2] == 0:
            link[1:3] = positive
        elif link[1] in remove and link[2] == 1 and nodes[link[3]]["type"] == "PreviewAny":
            remove.add(link[3])
    if all(link[3] in remove for link in graph["links"] if link[1] == config_id):
        remove.add(config_id)
    graph["nodes"] = [node for node in graph["nodes"] if node["id"] not in remove]
    graph["links"] = [link for link in graph["links"] if link[1] not in remove and link[3] not in remove]
    if config_graph is not None and config_id in remove:
        graph["definitions"]["subgraphs"] = [item for item in subgraphs if item is not config_graph]
    next_link = max(links) + 1
    for name, typ, origin in connections:
        hybrid["inputs"].append({"name": name, "type": typ, "shape": 7, "link": None})
        if origin is not None:
            graph["links"].append([next_link, *origin, hybrid["id"], len(hybrid["inputs"]) - 1, typ])
            next_link += 1
    # Rebuild socket backlinks after deleting adapters and reconnecting consumers.
    for node in graph["nodes"]:
        for port in node.get("inputs", []):
            port["link"] = None
        for port in node.get("outputs", []):
            port["links"] = []
    for lid, origin, oslot, target, tslot, _ in graph["links"]:
        nodes[origin]["outputs"][oslot]["links"].append(lid)
        nodes[target]["inputs"][tslot]["link"] = lid
    hybrid["title"] = "HYBRID WINDOWS — built-in video guides"
    hybrid["size"][1] = max(hybrid["size"][1], 680)
    for node in graph["nodes"]:
        if node["type"] == "MarkdownNote" and "Add Guide test" in node.get("title", ""):
            node["title"] = "Built-in hybrid guides"
            node["widgets_values"] = [
                "Control A feeds guide_video on H3 Hybrid Windows. Keep Ctrl A — LOOPS enabled. "
                "The hybrid stages align each source window and encode it at their own resolution, including after upscale.\n\n"
                "Keep the existing chunk prompt file, picture references, source preprocessing and resume controls. "
                "Describe the source as 'the guide video'. Both Basic Guiders use the Hybrid Windows positive output.\n\n"
                "Source audio remains a reference and uses the existing preservation masks. Optional guide_audio is separate.\n\n"
                "Output/resume uses the _hybrid_guides variant. Start fresh for comparison. "
                "Per-window frame ranges and guide resolutions are logged during sampling."
            ]
        if node["type"] == "StringConstant" and node.get("title") == "ADD GUIDE — output / resume variant":
            node["widgets_values"][0] = node["widgets_values"][0].removesuffix("_add_guide") + "_hybrid_guides"
            if "widgets_values_named" in node:
                node["widgets_values_named"]["string"] = node["widgets_values"][0]
            node["title"] = "HYBRID GUIDES — output / resume variant"
        if node["type"] in ("MMH3SaveResumeVideo", "H3SaveAVLatent"):
            node["widgets_values"] = [value.replace("add_guide/", "hybrid_guides/") if isinstance(value, str) else value
                                      for value in node.get("widgets_values", [])]
            if "filename_prefix" in node.get("widgets_values_named", {}):
                node["widgets_values_named"]["filename_prefix"] = node["widgets_values_named"]["filename_prefix"].replace("add_guide/", "hybrid_guides/")
    for group in graph.get("groups", []):
        if group.get("title", "").startswith("ADD GUIDE TEST"):
            group["title"] = "HYBRID GUIDES — one input for both sampling resolutions"
    graph.setdefault("extra", {}).pop("add_guide_test", None)
    graph["extra"]["integrated_guides"] = {"hybrid_node": hybrid["id"], "removed_nodes": sorted(remove)}
    # Give the new workflow independent subgraph identities.
    identities = {item["id"]: str(uuid.uuid4()) for item in graph.get("definitions", {}).get("subgraphs", [])}
    for item in graph.get("definitions", {}).get("subgraphs", []):
        item["id"] = identities[item["id"]]
    for item in [graph] + graph.get("definitions", {}).get("subgraphs", []):
        for node in item["nodes"]:
            node["type"] = identities.get(node["type"], node["type"])
    graph["id"] = str(uuid.uuid4())
    graph["revision"] = 0
    graph["last_link_id"] = next_link - 1
    return graph


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    raw = args.source.read_bytes()
    result = convert(json.loads(raw))
    result["extra"]["integrated_guides"].update(source_workflow=str(args.source), source_sha256=hashlib.sha256(raw).hexdigest())
    with args.destination.open("x") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(args.destination)
