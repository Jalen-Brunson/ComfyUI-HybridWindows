"""Plan keyframe positions with the same window geometry as the sampler."""

import math

from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import align_frame_count, video_latent_t

from .keyframes import MAX_KEYFRAMES, parse_indices
from .windows import AUDIO_PER_FRAME, frame_at, plan_windows


class H3HybridKeyframeCalculator(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3HybridKeyframeCalculator", display_name="H3 Hybrid Keyframe Calculator",
            category="sampling/hybrid",
            description=(
                "Calculate frame_indices for H3 Hybrid Keyframes without loading a model. "
                "Shows actual window ranges, shared keyframes and the final window's backward "
                "shift. Match the run length, window size and overlap used by Hybrid Windows."),
            inputs=[
                io.Int.Input("total_frames", default=175, min=5, max=1000000, step=17,
                             tooltip="Full run length. Rounded up to H3's 17k+5 frame grid, "
                                     "just like Empty MiniMax H3 AV Latent."),
                io.Int.Input("window_frames", default=56, min=22, max=3600, step=17,
                             tooltip="Use the same window_frames as H3 Hybrid Windows."),
                io.Int.Input("overlap_frames", default=5, min=5, max=3583, step=17,
                             tooltip="Use the same overlap_frames as H3 Hybrid Windows. "
                                     "The final overlap can be larger when the last window shifts back."),
                io.DynamicCombo.Input("placement", options=[
                    io.DynamicCombo.Option("evenly spaced", [
                        io.Int.Input("keyframe_count", default=4, min=1, max=MAX_KEYFRAMES,
                                     tooltip="Number of stills. Two or more include both the first "
                                             "and last frame; one uses frame 0."),
                    ]),
                    io.DynamicCombo.Option("window starts", []),
                    io.DynamicCombo.Option("custom", [
                        io.String.Input("frame_indices", default="0, 51, 102, 153",
                                        tooltip="One frame number per image, in socket order. "
                                                "Negative numbers count from the end: -1 is the last frame."),
                    ]),
                ], tooltip="Evenly spaced uses your image count. Window starts uses one image per "
                           "actual window, including any shifted final window. Custom checks your list."),
            ],
            outputs=[io.String.Output(display_name="frame_indices"),
                     io.Int.Output(display_name="total_frames"),
                     io.String.Output(display_name="report")],
        )

    @classmethod
    def execute(cls, total_frames, window_frames, overlap_frames, placement):
        total = align_frame_count(max(5, total_frames))
        geometry = plan_windows(window_frames, overlap_frames, 1)
        rows = video_latent_t(total)
        count = max(1, 1 + math.ceil((rows - geometry.length) / geometry.stride))
        plan = plan_windows(window_frames, overlap_frames, count)
        shapes = [(1, 1, rows, 1, 1), (1, 1, 1, round(total * AUDIO_PER_FRAME))]
        spans = [(frame_at(v0), frame_at(v1)) for v0, v1, _, _ in plan.spans(shapes)]

        mode = placement["placement"]
        if mode == "evenly spaced":
            keyframe_count = placement["keyframe_count"]
            if not 1 <= keyframe_count <= total:
                raise ValueError(f"Choose between 1 and {total} keyframes so every image has a unique frame.")
            indices = ([0] if keyframe_count == 1 else
                       [round(i * (total - 1) / (keyframe_count - 1)) for i in range(keyframe_count)])
        elif mode == "window starts":
            indices = [start for start, _ in spans]
        elif mode == "custom":
            indices = parse_indices(placement["frame_indices"])
            if not indices:
                raise ValueError("Enter at least one frame number in frame_indices.")
            indices = [index if index >= 0 else total + index for index in indices]
        else:
            raise ValueError(f"Unknown keyframe placement: {mode!r}.")
        seen = set()
        for frame in indices:
            if not 0 <= frame < total:
                raise ValueError(f"Keyframe {frame} is outside this run's frames 0..{total - 1}.")
            if frame in seen:
                raise ValueError(f"Frame {frame} is listed twice after resolving negative indices.")
            seen.add(frame)

        frame_indices = ", ".join(str(frame) for frame in indices)
        lines = [f"{len(indices)} keyframe(s): {frame_indices}",
                 f"Run: {total} frames (0..{total - 1}); {count} window(s) of "
                 f"{plan.window_frames} frames, {plan.overlap_frames} regular overlap."]
        if (total, plan.window_frames, plan.overlap_frames) != (total_frames, window_frames, overlap_frames):
            lines.append(f"H3 frame alignment: requested run/window/overlap "
                         f"{total_frames}/{window_frames}/{overlap_frames}; using "
                         f"{total}/{plan.window_frames}/{plan.overlap_frames}.")
        for i, (start, end) in enumerate(spans):
            overlap = f", {spans[i - 1][1] - start} frames shared with window {i}" if i else ""
            lines.append(f"  window {i + 1}: frames {start}..{end - 1} ({end - start} frames){overlap}")
        regular_start = frame_at((count - 1) * plan.stride)
        if spans[-1][0] < regular_start:
            lines.append(f"Final window shifted back {regular_start - spans[-1][0]} frames: "
                         f"starts at {spans[-1][0]} instead of {regular_start}, to end at {total - 1}.")
            lines.append(f"For {count} regularly spaced full windows, use {plan.total_frames} total frames.")
        for n, frame in enumerate(indices, 1):
            owners = [f"window {i + 1} at its frame {frame - start}"
                      for i, (start, end) in enumerate(spans) if start <= frame < end]
            lines.append(f"  keyframe {n}: frame {frame} -> " + " and ".join(owners)
                         + (" (shared overlap)" if len(owners) > 1 else ""))
        ordered = sorted(indices)
        separated = [(a, b) for a, b in zip(ordered, ordered[1:])
                     if not any(start <= a and b < end for start, end in spans)]
        if separated:
            pairs = ", ".join(f"{a}->{b}" for a, b in separated)
            lines.append(f"Transition context: no window sees both anchors for {pairs}. "
                         "Pinned history can still guide these transitions.")
        elif len(indices) > 1:
            lines.append("Transition context: every adjacent keyframe pair is visible together "
                         "in at least one window.")
        lines.append("Connect frame_indices to H3 Hybrid Keyframes; connect one image per listed frame.")
        report = "\n".join(lines)
        return io.NodeOutput(frame_indices, total, report)
