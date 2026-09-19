"""Still-image keyframes at frame numbers of the whole run, for the native chain.

Core's "Add Guide for MiniMax H3" anchors one image per node, at an index that
counts from the start of that conditioning's own window. On the hybrid chain
that means a chain of guide nodes per window and hand-converted frame numbers.

H3 Hybrid Keyframes takes every still and every frame number at once, counted
over the run's full timeline. H3 Hybrid Windows then works out which window
draws each frame (both windows when it falls in a shared overlap), converts
the number to that window's own index, fits the still to the canvas and
encodes it. The result is the same `minimax_keyframes` entry the core node
writes, so the model sees no difference and both stages scope it per window
exactly as they scope a core guide.
"""

import logging
import re

import comfy.utils
import nodes as core_nodes
from comfy_api.latest import io

from .windows import WINDOW_KEY, frame_at

KEYFRAMES_TYPE = io.Custom("H3_HYBRID_KEYFRAMES")
FIT_MODES = ["crop", "stretch"]
MAX_KEYFRAMES = 64
_SEPARATORS = re.compile(r"[,;\s]+")


def parse_indices(text):
    """'0, 15 32;64\\n-1' -> [0, 15, 32, 64, -1]; any other token is named in the error."""
    out = []
    for token in _SEPARATORS.split(text or ""):
        if not token:
            continue
        try:
            out.append(int(token))
        except ValueError:
            raise ValueError(
                f"frame_indices: {token!r} is not a whole frame number. List one frame number per "
                "image, separated by commas, e.g. '0, 15, 32, 64, 100'. Negative numbers count from "
                "the end (-1 = the last frame).") from None
    return out


def fit_still(still, width, height, fit):
    """[1, H, W, C] -> [1, height, width, 3], the way core's guide node resizes."""
    samples = still[..., :3].movedim(-1, 1)
    crop = "center" if fit == "crop" else "disabled"
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


class H3HybridKeyframes(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3HybridKeyframes", display_name="H3 Hybrid Keyframes",
            category="sampling/hybrid", is_experimental=True,
            description=(
                "Stills pinned at frame numbers of the whole run. Plug each image into its own "
                "socket (a new socket appears as you connect one), list the frame numbers in the "
                "same order, and connect the output to the `keyframes` input of H3 Hybrid Windows. "
                "No per-window arithmetic: numbers count over the full timeline from 0, negative "
                "from the end."),
            inputs=[
                io.Vae.Input("vae", tooltip="The H3 video VAE, the same one the window encoders use."),
                io.String.Input(
                    "frame_indices", default="0", multiline=False,
                    tooltip="One frame number per image, in socket order, e.g. '0, 15, 32, 64, 100'. "
                            "Counted over the WHOLE run from 0 (24 fps unless the timeline says "
                            "otherwise). Negative numbers count from the end: -1 is the last frame. "
                            "A socket carrying a batch uses one number per frame of the batch."),
                io.Autogrow.Input("images", template=io.Autogrow.TemplatePrefix(
                    input=io.Image.Input("keyframe"), prefix="keyframe_", min=1, max=MAX_KEYFRAMES)),
                io.Combo.Input(
                    "fit", options=FIT_MODES, default="crop", optional=True,
                    tooltip="How a still with another aspect is fitted to the canvas: crop keeps the "
                            "aspect and trims the edges (core's Add Guide behaviour); stretch keeps "
                            "every pixel and distorts."),
                io.Int.Input(
                    "width", default=0, min=0, max=core_nodes.MAX_RESOLUTION, step=32, optional=True,
                    tooltip="Canvas width. Leave 0 to take it from the run: the master latent connected "
                            "to H3 Hybrid Windows, or the encoders' own first/last guides. Set both "
                            "width and height when neither exists (plain prompts, no master latent)."),
                io.Int.Input(
                    "height", default=0, min=0, max=core_nodes.MAX_RESOLUTION, step=32, optional=True,
                    tooltip="Canvas height; see width."),
            ],
            outputs=[KEYFRAMES_TYPE.Output(display_name="keyframes"),
                     io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, vae, frame_indices, images=None, fit="crop", width=0, height=0):
        stills = []
        for name in sorted(images or {}, key=lambda n: int(n.rsplit("_", 1)[1])):
            batch = images[name]
            if batch is None:
                continue
            if getattr(batch, "ndim", 0) != 4:
                raise ValueError(f"{name}: expected an IMAGE [batch, height, width, channels].")
            for i in range(batch.shape[0]):
                stills.append((name, batch[i:i + 1]))
        if not stills:
            raise ValueError("Connect at least one image to H3 Hybrid Keyframes.")
        indices = parse_indices(frame_indices)
        if len(indices) != len(stills):
            raise ValueError(
                f"H3 Hybrid Keyframes: {len(stills)} image(s) but {len(indices)} frame number(s) in "
                f"{frame_indices!r}. Give one frame number per image, in socket order.")
        if (int(width) > 0) != (int(height) > 0):
            raise ValueError("H3 Hybrid Keyframes: set both width and height, or leave both at 0.")
        seen = {}
        for n, index in enumerate(indices, 1):
            if index in seen:
                raise ValueError(f"H3 Hybrid Keyframes: frame {index} is listed twice "
                                 f"(images {seen[index]} and {n}).")
            seen[index] = n
        if fit not in FIT_MODES:
            raise ValueError(f"H3 Hybrid Keyframes: fit must be one of {FIT_MODES}.")
        lines = [f"{len(stills)} keyframe(s), fit by {fit}; frame numbers count over the whole run"
                 + (f"; canvas {int(width)}x{int(height)}." if int(width) else
                    "; canvas taken from the run (master latent or the encoders' guides).")]
        for n, ((name, still), index) in enumerate(zip(stills, indices), 1):
            where = f"frame {index}" if index >= 0 else f"frame {index} (from the end)"
            lines.append(f"  keyframe {n} ({name}, {still.shape[2]}x{still.shape[1]}): {where}")
        lines.append("Connect to H3 Hybrid Windows' `keyframes` input; its report shows the window "
                     "each frame landed in.")
        payload = {"stills": stills, "indices": indices, "vae": vae, "fit": fit,
                   "width": int(width), "height": int(height)}
        return io.NodeOutput(payload, "\n".join(lines))


def _canvas(keyframes, bound, prepared):
    """(width, height, where it came from), or a ValueError that says what to connect."""
    width, height = keyframes.get("width", 0), keyframes.get("height", 0)
    if width and height:
        return int(width), int(height), "set on H3 Hybrid Keyframes"
    if prepared is not None:
        video = prepared["samples"].unbind()[0]
        return int(video.shape[4]) * 16, int(video.shape[3]) * 16, "the master latent"
    for _, metadata in bound:
        for guide in metadata.get("minimax_keyframes", []):
            latent = guide.get("latent")
            if latent is not None:
                return int(latent.shape[4]) * 16, int(latent.shape[3]) * 16, "the encoders' own guide"
    raise ValueError(
        "H3 Hybrid Keyframes cannot tell the canvas size. Set its width and height to the "
        "generation size (the same values the window encoders use), or connect the master "
        "latent to H3 Hybrid Windows.")


def _window_spans(plan, count, prepared):
    """Per-window (first frame, end frame) in pixel frames, and the run's frame count."""
    if prepared is not None:
        video, audio = prepared["samples"].unbind()
        spans = plan.spans([tuple(video.shape), tuple(audio.shape)])
        return [(frame_at(v0), frame_at(v1)) for v0, v1, _, _ in spans], frame_at(video.shape[2])
    spans = [(frame_at(i * plan.stride), frame_at(i * plan.stride + plan.length)) for i in range(count)]
    return spans, plan.total_frames


def attach_keyframes(keyframes, bound, plan, count, prepared, accepted_v=0):
    """Encode each still into the bound conditioning of every window that draws its frame.

    `bound` is the per-window list H3 Hybrid Windows built (each entry tagged with
    WINDOW_KEY). Returns the report lines; raises on a frame outside the run, a
    frame named twice, or a canvas nobody can determine.
    """
    if keyframes is None:
        return []
    stills, indices = keyframes["stills"], keyframes["indices"]
    vae, fit = keyframes["vae"], keyframes.get("fit", "crop")
    width, height, source = _canvas(keyframes, bound, prepared)
    spans, total = _window_spans(plan, count, prepared)
    accepted_frames = frame_at(accepted_v) if accepted_v else 0
    lines = [f"Keyframes: {len(stills)} still(s) on the {width}x{height} canvas (size from {source}); "
             f"frame numbers count over this run's {total} frames."]
    resolved = {}
    for n, ((socket, still), index) in enumerate(zip(stills, indices), 1):
        frame = index if index >= 0 else total + index
        if not 0 <= frame < total:
            raise ValueError(
                f"keyframe {n} ({socket}): frame {index} is outside this run's frames 0..{total - 1} "
                "(negative numbers count from the end).")
        if frame in resolved:
            raise ValueError(f"keyframe {n} ({socket}) and keyframe {resolved[frame]} both name frame {frame}.")
        resolved[frame] = n
        owners = [k for k, (start, end) in enumerate(spans) if start <= frame < end]
        if not owners:  # a short master whose windows slid back; the plan cannot leave gaps otherwise
            raise ValueError(f"keyframe {n} ({socket}): no window draws frame {frame}.")
        fitted = fit_still(still, width, height, fit)
        latent = vae.encode(fitted)
        placed = []
        for k in owners:
            local = frame - spans[k][0]
            replaced = False
            for _, metadata in bound:
                if metadata.get(WINDOW_KEY) != k:
                    continue
                guides = [dict(g) for g in metadata.get("minimax_keyframes", [])]
                hit = next((g for g in guides if g.get("resolved_frame_index") == local
                            and g.get("latent") is not None), None)
                if hit is not None:
                    hit["latent"] = latent
                    replaced = True
                else:
                    guides.append({"resolved_frame_index": local, "latent": latent})
                metadata["minimax_keyframes"] = guides
            placed.append(f"window {k + 1} at its frame {local}" + (" (replaces the encoder's guide there)" if replaced else ""))
        note = ""
        if (still.shape[2], still.shape[1]) != (width, height):
            note = f", fitted by {fit} from {still.shape[2]}x{still.shape[1]}"
        if frame < accepted_frames:
            note += ", inside the accepted prefix so it changes nothing"
        joiner = " and " if len(placed) == 2 else ", "
        lines.append(f"  keyframe {n} ({socket}): frame {frame} -> " + joiner.join(placed)
                     + (" (shared overlap)" if len(placed) > 1 else "") + note)
    for line in lines:
        logging.info("[Hybrid Windows] %s", line.strip())
    return lines
