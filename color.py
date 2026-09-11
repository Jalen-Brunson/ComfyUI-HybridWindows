"""Temporal chroma stabilization of decoded RGB video frames."""

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d, median_filter

import comfy.model_management
from comfy.utils import ProgressBar
from comfy_api.latest import io


def color_stats(images, progress):
    stats = np.empty((len(images), 3), dtype=np.float64)
    for i, image in enumerate(images):
        comfy.model_management.throw_exception_if_processing_interrupted()
        rgb = image.detach().to(device="cpu", dtype=torch.float32).numpy()
        y = .2126 * rgb[..., 0] + .7152 * rgb[..., 1] + .0722 * rgb[..., 2]
        cb, cr = (rgb[..., 2] - y) / 1.8556, (rgb[..., 0] - y) / 1.5748
        stats[i] = cb.mean(), cr.mean(), np.hypot(cb.std(), cr.std())
        progress.update(1)
    return stats


def smooth_colors(stats, fps, seconds):
    if len(stats) < 3:
        return stats.copy()
    radius = max(1, round(.25 * fps))
    smoothed = median_filter(stats, size=(2 * radius + 1, 1), mode="nearest")
    sigma = seconds * fps
    if sigma > .5:
        smoothed = gaussian_filter1d(smoothed, sigma=sigma, axis=0, mode="nearest")
    return smoothed


def color_plan(stats, source_stats, fps, reference_start, reference_end, smoothing, strength,
               max_tint_shift=.06):
    count = len(stats)
    start = max(0, min(round(reference_start * fps), count - 1))
    end = max(start + 1, min(round(reference_end * fps), count))
    anchor = np.median(stats[start:end], axis=0)
    smoothed = smooth_colors(stats, fps, smoothing)
    target = np.broadcast_to(anchor, stats.shape).copy()
    if source_stats is not None:
        source = smooth_colors(source_stats, fps, smoothing)
        source_anchor = np.median(source_stats[start:end], axis=0)
        target[:, :2] += source[:, :2] - source_anchor[:2]
        # Different subjects can have different color strength. Source variation
        # may raise this target, but does not lower it below the generated anchor.
        target[:, 2] *= np.clip(source[:, 2] / max(source_anchor[2], 1e-6), 1., 4.)
    frame = np.arange(count)
    ramp_frames = max(1., 2. * fps)
    ramp = np.maximum(np.clip((frame - (end - 1)) / ramp_frames, 0., 1.),
                      np.clip((start - frame) / ramp_frames, 0., 1.))
    amount = strength * ramp
    gain = 1. + amount * (np.clip(target[:, 2] / np.maximum(smoothed[:, 2], 1e-6), .9, 1.4) - 1.)
    shift = amount[:, None] * np.clip(target[:, :2] - smoothed[:, :2], -max_tint_shift, max_tint_shift)
    return smoothed[:, :2], gain, shift


def correct_frame(rgb, center, gain, shift):
    y = .2126 * rgb[..., 0] + .7152 * rgb[..., 1] + .0722 * rgb[..., 2]
    cb, cr = (rgb[..., 2] - y) / 1.8556, (rgb[..., 0] - y) / 1.5748
    dcb = (cb - center[0]) * (gain - 1.) + shift[0]
    dcr = (cr - center[1]) * (gain - 1.) + shift[1]
    delta = np.empty_like(rgb)
    delta[..., 0] = 1.5748 * dcr
    delta[..., 2] = 1.8556 * dcb
    delta[..., 1] = -(.2126 * delta[..., 0] + .0722 * delta[..., 2]) / .7152
    # Limit the color change at the RGB gamut boundary, preserving luma rather
    # than clipping channels independently and moving highlights/shadows.
    room = np.where(delta > 0, 1. - rgb, rgb)
    magnitude = np.abs(delta)
    limits = np.ones_like(delta)
    np.divide(room, magnitude, out=limits, where=magnitude > 1e-12)
    scale = np.minimum(1., np.min(limits, axis=-1))
    return np.clip(rgb + delta * scale[..., None], 0., 1.)


class VideoColorStabilize(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="VideoColorStabilize", display_name="Video Color Stabilize",
            category="image/video",
            description="Smooth tint and saturation drift in SDR/sRGB frames using an opening reference interval. "
                        "Preserves brightness. Connect after video decoding and before Create/Save Video. "
                        "Supply the complete continuous shot as one frame batch.",
            inputs=[
                io.Image.Input("images"),
                io.Boolean.Input("enabled", default=True),
                io.Float.Input("strength", default=.85, min=0., max=1., step=.05),
                io.Float.Input("fps", default=24., min=.001, max=240., step=1.,
                               tooltip="Match the output video's frame rate."),
                io.Float.Input("reference_start_seconds", default=1., min=0., max=3600., step=.1),
                io.Float.Input("reference_end_seconds", default=8., min=0., max=3600., step=.1,
                               tooltip="This interval is left unchanged; correction ramps in over two seconds outside it."),
                io.Float.Input("smoothing_seconds", default=2., min=0., max=30., step=.1,
                               tooltip="Smooth the measured color drift over time, without blurring frames."),
                io.Image.Input("source_frames", optional=True,
                               tooltip="Optional original source frames, before added noise or color effects. "
                                       "Must have the same frame count and timing as images. "
                                       "Leave disconnected for image-conditioned generation."),
                io.Float.Input("max_tint_shift", default=.06, min=0., max=.5, step=.005,
                               optional=True,
                               tooltip="Maximum Cb/Cr offset before strength is applied. "
                                       "0.06 allows about 15 levels on a 0–255 scale. "
                                       "A lower limit can leave strong drift uncorrected. "
                                       "Use strength 1 for full correction toward the source-relative target."),
            ],
            outputs=[io.Image.Output(display_name="images")],
        )

    @classmethod
    def execute(cls, images, enabled=True, strength=.85, fps=24., reference_start_seconds=1.,
                reference_end_seconds=8., smoothing_seconds=2., source_frames=None, max_tint_shift=.06):
        if not enabled or strength == 0 or len(images) == 0:
            return io.NodeOutput(images)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("Video Color Stabilize needs an RGB IMAGE batch from the video decoder.")
        if fps <= 0 or reference_end_seconds <= reference_start_seconds:
            raise ValueError("Set a positive fps and a reference end later than its start.")
        if source_frames is not None:
            if source_frames.ndim != 4 or source_frames.shape[-1] != 3 or len(source_frames) != len(images):
                raise ValueError("Video Color Stabilize: source_frames must be RGB with the same frame count "
                                 "as images. Trim and align the source to the generated timeline first.")
        progress = ProgressBar(len(images) * (3 if source_frames is not None else 2))
        stats = color_stats(images, progress)
        source_stats = color_stats(source_frames, progress) if source_frames is not None else None
        centers, gains, shifts = color_plan(stats, source_stats, fps, reference_start_seconds,
                                           reference_end_seconds, smoothing_seconds, strength, max_tint_shift)
        output = torch.empty_like(images)
        for i, image in enumerate(images):
            comfy.model_management.throw_exception_if_processing_interrupted()
            if gains[i] == 1. and not np.any(shifts[i]):
                output[i].copy_(image)
            else:
                rgb = image.detach().to(device="cpu", dtype=torch.float32).numpy()
                corrected = correct_frame(rgb, centers[i], gains[i], shifts[i])
                output[i].copy_(torch.from_numpy(corrected))
            progress.update(1)
        return io.NodeOutput(output)
