"""Derive the native-sampler edition of the 05 hybrid resume workflow.

The 05 flow drives one MMH3HybridWindowSampler. This writes a sibling that
drives the same inputs through H3HybridWindows and two stock
SamplerCustomAdvanced nodes, and changes nothing else: the same model chain,
cond set, masks, resume plumbing, merge/save and output nodes, wire for wire.

Always derived from the 05 file, never from its own output, so re-running is
idempotent and a hand edit to 05N is replaced rather than merged. The 05 file
itself is only read.

    python3 derive_05n_workflow.py [--dest PATH] [--check]
"""

import argparse
import asyncio
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMFY = ROOT.parents[1]
sys.path.insert(0, str(COMFY))
USER_ARGS = sys.argv[1:]
sys.argv = [sys.argv[0], "--cpu"]  # comfy parses argv at import
import comfy.options
comfy.options.enable_args_parsing()
import nodes
from comfy_extras import nodes_custom_sampler


async def register():
    """Node schemas come from this ComfyUI, so new sockets appear by themselves."""
    extension = await nodes_custom_sampler.comfy_entrypoint()
    for cls in await extension.get_node_list():
        nodes.NODE_CLASS_MAPPINGS[cls.GET_SCHEMA().node_id] = cls
    assert await nodes.load_custom_node(str(ROOT))

SOURCE = COMFY / "user/default/workflows/H3 Diagnostics/05 Hybrid - preserve accepted prefix.json"
DEST = SOURCE.parent / "05N Hybrid NATIVE - preserve accepted prefix.json"
SAMPLER_ID = 306          # the node this replaces
GROUP = [3140, -1500, 2400, 1000]   # empty canvas above the existing SAMPLING group
SPLIT_STEP = 6            # sequential steps; the rest run jointly
TOTAL_STEPS = 8           # the PDD Acc distill's schedule

NOTE = """## Native hybrid chain

This is the 05 resume flow with the single hybrid sampler replaced by the
published native chain. Everything else -- model, cond set, masks, resume,
merge, save, grading -- is the 05 wiring untouched.

**H3 Hybrid Windows** plans the windows and hands out two models. It also
prepares the master: it composes the inpaint masks and the accepted prefix into
one per-row noise mask, so **sample its `latent` output**, not the master
directly.

**1. SEQUENTIAL warm-up** runs the windows in order with the overlap carried,
stopping at the split. **2. JOINT finish** completes them together from the
state the warm-up left. Take its `output` (slot 0), never `denoised_output`.

- `Switch at step` on Split Sigmas is the warm-up's step count. It must match
  `total_steps` on the windows node and the PDD distill's `nfe` (8 = 6+2).
- `overlap_pin = tail_custom`, `context_frames = 5` pins the last 5 video frames
  of the full 39-frame overlap during warmup. The other 34 can regenerate
  internally; accepted output, audio carry and the joint finish stay unchanged.
  Partial pinning requires full-frame regeneration outside the accepted prefix.
  Use `overlap_pin = full` for an inpaint mask protecting fresh source video.
- `noise_mode = per_window` draws seed + resume shift + window index per
  window, the same draws the single-node sampler makes, so a run can be
  compared with one made by 05.
- Both stages must run in one queue: the joint stage resumes the warm-up's
  captured state, and a restart between them loses it (it says so rather than
  sampling something wrong).
- Per-chunk mp4 previews are not available here (they hook a GUIDER the
  looping sampler calls once per chunk); step previews work normally."""


def node_from_schema(kind, node_id, title, pos, size, values=None):
    """One node in this file's shape, from the live schema."""
    cls = nodes.NODE_CLASS_MAPPINGS[kind]
    info = (cls.GET_NODE_INFO_V1() if hasattr(cls, "GET_NODE_INFO_V1") else
            {"input": cls.INPUT_TYPES(), "output": cls.RETURN_TYPES,
             "output_name": getattr(cls, "RETURN_NAMES", cls.RETURN_TYPES)})
    values = values or {}
    sockets, widgets, named = [], [], {}
    for section in ("required", "optional"):
        for name, definition in info["input"].get(section, {}).items():
            typ = definition[0]
            options = definition[1] if len(definition) > 1 else {}
            if typ == "COMFY_AUTOGROW_V3":
                continue  # driven by cond_set here; no member sockets
            socket = {"localized_name": name, "name": name,
                      "type": "COMBO" if isinstance(typ, list) else typ, "link": None}
            if section == "optional":
                socket["shape"] = 7
            if isinstance(typ, list) or typ in ("INT", "FLOAT", "STRING", "BOOLEAN", "COMBO"):
                socket["widget"] = {"name": name}
                choices = typ if isinstance(typ, list) else options.get("options", [])
                default = options.get("default", choices[0] if choices else
                                      (False if typ == "BOOLEAN" else "" if typ == "STRING" else 0))
                value = values.get(name, default)
                widgets.append(value)
                named[name] = value
                if options.get("control_after_generate"):
                    widgets.append("fixed")
            sockets.append(socket)
    assert not set(values) - {s["name"] for s in sockets}, (kind, set(values))
    sockets.sort(key=lambda socket: "widget" in socket)  # connections before widgets
    properties = {"Node name for S&R": kind}
    if kind not in ("H3HybridWindows", "H3HybridControlNet", "VideoColorStabilize"):
        properties["cnr_id"] = "comfy-core"
    return {"id": node_id, "type": kind, "pos": list(pos), "size": list(size), "flags": {},
            "order": node_id, "mode": 0, "inputs": sockets,
            "outputs": [{"localized_name": label, "name": label, "type": typ, "links": []}
                        for label, typ in zip(info["output_name"], info["output"])],
            "title": title, "properties": properties,
            "widgets_values": widgets, "widgets_values_named": named}


class Graph:
    """The 05 graph, with the link table kept consistent on both ends."""

    def __init__(self, data):
        self.data = data
        self.nodes = {n["id"]: n for n in data["nodes"]}
        self.links = {l[0]: l for l in data["links"]}

    def slot(self, node_id, name):
        return next(i for i, s in enumerate(self.nodes[node_id]["inputs"]) if s["name"] == name)

    def out_slot(self, node_id, name):
        return next(i for i, s in enumerate(self.nodes[node_id]["outputs"]) if s["name"] == name)

    def add_node(self, node):
        self.data["nodes"].append(node)
        self.nodes[node["id"]] = node
        return node["id"]

    def retarget(self, link_id, node_id, name):
        """Point an existing link at a new consumer, keeping its id and source."""
        link = self.links[link_id]
        slot = self.slot(node_id, name)
        link[3], link[4] = node_id, slot
        self.nodes[node_id]["inputs"][slot]["link"] = link_id
        return link_id

    def resource(self, link_id, node_id, name):
        """Point an existing link at a new producer, keeping its id and target."""
        link = self.links[link_id]
        slot = self.out_slot(node_id, name)
        link[1], link[2] = node_id, slot
        self.nodes[node_id]["outputs"][slot]["links"].append(link_id)
        return link_id

    def connect(self, source, source_name, target, target_name):
        link_id = self.data["last_link_id"] + 1
        self.data["last_link_id"] = link_id
        source_slot, target_slot = self.out_slot(source, source_name), self.slot(target, target_name)
        typ = self.nodes[source]["outputs"][source_slot]["type"]
        link = [link_id, source, source_slot, target, target_slot, typ]
        self.data["links"].append(link)
        self.links[link_id] = link
        self.nodes[source]["outputs"][source_slot]["links"].append(link_id)
        self.nodes[target]["inputs"][target_slot]["link"] = link_id
        return link_id

    def drop_node(self, node_id):
        node = self.nodes.pop(node_id)
        self.data["nodes"] = [n for n in self.data["nodes"] if n["id"] != node_id]
        incoming = {s["name"]: s["link"] for s in node["inputs"] if s.get("link") is not None}
        outgoing = {o["name"]: list(o.get("links") or []) for o in node["outputs"]}
        return incoming, outgoing


def rectangles_free(graph, box):
    """Nothing may already live where the new group goes."""
    def hits(a, b):
        return a[0] < b[0]+b[2] and b[0] < a[0]+a[2] and a[1] < b[1]+b[3] and b[1] < a[1]+a[3]
    for node in graph.data["nodes"]:
        pos, size = node["pos"], node.get("size", [200, 60])
        if hits(box, [pos[0], pos[1]-30, size[0], size[1]+30]):
            return False, f"node {node['id']} ({node['type']})"
    for group in graph.data.get("groups", []):
        if hits(box, group["bounding"]):
            return False, f"group {group.get('title')}"
    return True, None


def derive():
    if "H3HybridWindows" not in nodes.NODE_CLASS_MAPPINGS:
        asyncio.run(register())  # already awaited when a caller runs inside a loop
    graph = Graph(json.loads(SOURCE.read_text()))
    free, blocker = rectangles_free(graph, GROUP)
    assert free, f"the new sampling group would overlap {blocker}"
    incoming, outgoing = graph.drop_node(SAMPLER_ID)

    first = graph.data["last_node_id"] + 1
    windows_id = graph.add_node(node_from_schema(
        "H3HybridWindows", first,
        "HYBRID WINDOWS (native) — plan · masks · accepted prefix · per-window noise",
        (3160, -1440), (470, 470),
        {"window_frames": 243, "overlap_frames": 39, "total_steps": TOTAL_STEPS,
         "denoise_mask_mode": "max", "accepted_prefix_frames": 0, "start_window": 0,
         "noise_mode": "per_window", "overlap_pin": "tail_custom", "context_frames": 5}))
    split_id = graph.add_node(node_from_schema(
        "SplitSigmas", first + 1, f"Switch at step {SPLIT_STEP} — PDD Acc sigmas",
        (3700, -1440), (300, 100), {"step": SPLIT_STEP}))
    guider_seq = graph.add_node(node_from_schema(
        "BasicGuider", first + 2, "Guider 1 — sequential model", (3700, -1300), (340, 80)))
    guider_joint = graph.add_node(node_from_schema(
        "BasicGuider", first + 3, "Guider 2 — joint model", (3700, -1170), (340, 80)))
    disable_id = graph.add_node(node_from_schema(
        "DisableNoise", first + 4, "Joint stage: no fresh noise", (3700, -1040), (250, 60)))
    warm_id = graph.add_node(node_from_schema(
        "SamplerCustomAdvanced", first + 5,
        f"1. SEQUENTIAL warm-up — steps 0..{SPLIT_STEP}, overlap carried",
        (4120, -1440), (360, 190)))
    joint_id = graph.add_node(node_from_schema(
        "SamplerCustomAdvanced", first + 6,
        f"2. JOINT finish — steps {SPLIT_STEP}..{TOTAL_STEPS} from the warm-up's state",
        (4560, -1440), (360, 190)))
    note_id = first + 7
    graph.add_node({"id": note_id, "type": "MarkdownNote", "pos": [4120, -1200], "size": [800, 640],
                    "flags": {}, "order": note_id, "mode": 0, "inputs": [], "outputs": [],
                    "title": "READ ME — native hybrid chain", "properties": {},
                    "widgets_values": [NOTE], "widgets_values_named": {"text": NOTE},
                    "color": "#222", "bgcolor": "#000"})

    # The sampler's own inputs move to the windows node, unchanged.
    for name in ("model", "cond_set", "latent", "denoise_mask", "audio_denoise_mask",
                 "window_frames", "overlap_frames", "accepted_prefix_frames", "start_window"):
        graph.retarget(incoming[name], windows_id, name)
    graph.retarget(incoming["sigmas"], split_id, "sigmas")
    graph.retarget(incoming["noise"], warm_id, "noise")
    graph.retarget(incoming["sampler"], warm_id, "sampler")
    # The second sampler shares the same euler selector.
    sampler_source = graph.links[incoming["sampler"]][1]
    graph.connect(sampler_source, graph.nodes[sampler_source]["outputs"][0]["name"], joint_id, "sampler")

    graph.connect(windows_id, "sequential_model", guider_seq, "model")
    graph.connect(windows_id, "positive", guider_seq, "conditioning")
    graph.connect(windows_id, "joint_model", guider_joint, "model")
    graph.connect(windows_id, "positive", guider_joint, "conditioning")
    graph.connect(windows_id, "latent", warm_id, "latent_image")
    graph.connect(guider_seq, "GUIDER", warm_id, "guider")
    graph.connect(split_id, "high_sigmas", warm_id, "sigmas")
    graph.connect(disable_id, "NOISE", joint_id, "noise")
    graph.connect(guider_joint, "GUIDER", joint_id, "guider")
    graph.connect(split_id, "low_sigmas", joint_id, "sigmas")
    graph.connect(warm_id, "output", joint_id, "latent_image")

    # Everything downstream now reads the joint stage; the report comes from the node.
    for link_id in outgoing["latent"]:
        graph.resource(link_id, joint_id, "output")
    for link_id in outgoing["report"]:
        graph.resource(link_id, windows_id, "report")

    graph.data["last_node_id"] = note_id
    graph.data["groups"] = list(graph.data.get("groups", [])) + [{
        "id": max([g.get("id", 0) for g in graph.data.get("groups", [])] + [0]) + 1,
        "title": "SAMPLING (NATIVE) — H3 Hybrid Windows → 2× SamplerCustomAdvanced",
        "bounding": list(GROUP), "color": "#3f789e", "flags": {}}]
    graph.data["id"] = str(uuid.uuid5(uuid.NAMESPACE_URL, "https://github.com/Jalen-Brunson/"
                                      "ComfyUI-HybridWindows/05N-native-resume"))
    return graph


async def api_schema_check():
    """The node must validate with only a cond set wired, no prompt sockets.

    Needs a node that emits an MMH3_COND_SET, so it reports "skipped" where
    MMH3Tools is not installed rather than failing on its absence.
    """
    import execution
    if "MMH3CondToSet" not in nodes.NODE_CLASS_MAPPINGS:
        sibling = ROOT.parent / "ComfyUI-MMH3Tools"
        if sibling.is_dir():
            try:
                await nodes.load_custom_node(str(sibling))
            except Exception:
                pass
    if "MMH3CondToSet" not in nodes.NODE_CLASS_MAPPINGS:
        return None, "skipped: MMH3Tools is not installed, so no MMH3_COND_SET producer exists"
    prompt = {
        "1": {"class_type": "H3HybridWindows",
              "inputs": {"model": ["2", 0], "window_frames": 243, "overlap_frames": 39,
                         "total_steps": TOTAL_STEPS, "noise_mode": "per_window",
                         "cond_set": ["3", 0]}},
        "2": {"class_type": "UNETLoader",
              "inputs": {"unet_name": nodes.NODE_CLASS_MAPPINGS["UNETLoader"].INPUT_TYPES()
                         ["required"]["unet_name"][0][0], "weight_dtype": "default"}},
        "3": {"class_type": "MMH3CondToSet", "inputs": {"conditioning": ["4", 0], "count": 3}},
        "4": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["5", 0]}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"text": "", "clip": ["6", 0]}},
        # SaveLatent is an output node in the base registry, so the check needs
        # no comfy_extras beyond the sampler module already registered.
        "7": {"class_type": "SaveLatent",
              "inputs": {"samples": ["1", 4], "filename_prefix": "05n_schema_check"}},
        "6": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": nodes.NODE_CLASS_MAPPINGS["CLIPLoader"].INPUT_TYPES()
                         ["required"]["clip_name"][0][0], "type": "minimax"}},
    }
    valid, error, *_ = await execution.validate_prompt("05n-schema-check", prompt, None)
    return valid, error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", default=str(DEST))
    parser.add_argument("--check", action="store_true", help="derive and report, write nothing")
    args = parser.parse_args(USER_ARGS)
    graph = derive()
    text = json.dumps(graph.data, indent=2) + "\n"
    report = {"nodes": len(graph.data["nodes"]), "links": len(graph.data["links"]),
              "last_node_id": graph.data["last_node_id"],
              "last_link_id": graph.data["last_link_id"]}
    if not args.check:
        Path(args.dest).write_text(text)
        report["written"] = args.dest
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
