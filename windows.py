"""H3 window geometry and prediction fusion for native ComfyUI sampling."""

from dataclasses import dataclass

import torch

import comfy.conds
import comfy.model_management
import comfy.utils
from comfy.context_windows import ContextHandlerABC, create_weights_pyramid
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_extras.nodes_minimax_h3 import align_frame_count, video_latent_t

WINDOW_KEY = "hybrid_window_index"
OFFSET_KEY = "hybrid_window_frame0"


def frame_at(index):
    groups, remainder = divmod(index, len(FRAME_PER_TOKEN))
    return groups * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:remainder])


@dataclass(frozen=True)
class WindowPlan:
    window_frames: int
    overlap_frames: int
    count: int

    def __post_init__(self):
        if self.count < 1:
            raise ValueError("Connect at least one window's conditioning.")
        if self.overlap_frames >= self.window_frames:
            raise ValueError("Overlap must be smaller than the window.")

    @property
    def length(self):
        return video_latent_t(self.window_frames)

    @property
    def stride(self):
        return self.length - video_latent_t(self.overlap_frames)

    @property
    def total_frames(self):
        return self.window_frames + (self.count - 1) * (self.window_frames - self.overlap_frames)

    def spans(self, shapes):
        video, audio = shapes
        expected = video_latent_t(self.total_frames)
        if video[2] != expected or audio[3] != round(self.total_frames * 40 / 24):
            raise ValueError(
                f"Hybrid windows need {self.total_frames} frames for {self.count} prompts. "
                "Connect total_frames to the Empty MiniMax H3 AV Latent length.")
        spans = []
        for index in range(self.count):
            v0 = index * self.stride
            v1 = v0 + self.length
            a0 = round(frame_at(v0) * 40 / 24)
            a1 = audio[3] if v1 == expected else round(frame_at(v1) * 40 / 24)
            spans.append((v0, v1, a0, a1))
        return spans


def plan_windows(window_frames, overlap_frames, count):
    return WindowPlan(align_frame_count(max(5, window_frames)),
                      align_frame_count(max(5, overlap_frames)), count)


def slice_av(parts, span):
    v0, v1, a0, a1 = span
    return [parts[0][:, :, v0:v1], parts[1][:, :, :, a0:a1]]


def write_new(destination, source, span, previous):
    v0, v1, a0, a1 = span
    pv, pa = previous
    first_v, first_a = max(v0, pv), max(a0, pa)
    destination[0][:, :, first_v:v1] = source[0][:, :, first_v-v0:].to(destination[0])
    destination[1][:, :, :, first_a:a1] = source[1][:, :, :, first_a-a0:].to(destination[1])


def select_conditioning(conds, index):
    if conds is None:
        return None
    selected = [c for c in conds if c.get(WINDOW_KEY, index) == index]
    if not selected:
        raise ValueError(f"No conditioning for hybrid window {index + 1}.")
    return selected


def window_options(options, span):
    return {**options, "transformer_options": {
        **options.get("transformer_options", {}), OFFSET_KEY: frame_at(span[0]),
    }}


class JointWindows(ContextHandlerABC):
    """Fuse clean predictions, then let the stock sampler take one global step."""

    def __init__(self, plan):
        self.plan = plan

    def should_use_context(self, model, conds, x_in, timestep, model_options):
        # Even a single window must select its tagged conditioning.
        return True

    def get_resized_cond(self, cond_in, x_in, window, device=None):
        index, span, shapes = window
        selected = select_conditioning(cond_in, index)
        if selected is None:
            return None
        result = []
        v0, v1, a0, a1 = span
        for cond in selected:
            model_conds = dict(cond["model_conds"])
            model_conds["latent_shapes"] = comfy.conds.CONDConstant(shapes)
            for key, dim, start, end in (("denoise_mask", 2, v0, v1),
                                         ("audio_denoise_mask", 3, a0, a1)):
                if key in model_conds:
                    value = model_conds[key]
                    model_conds[key] = value._copy_with(value.cond.narrow(dim, start, end-start))
            # Text, tags, references and window-local guides stay as encoded.
            # Native H3 rebuilds its packed layout for the window's new shapes.
            result.append({**cond, "model_conds": model_conds})
        return result

    def execute(self, calc_cond_batch, model, conds, x_in, timestep, model_options):
        shapes = next(c["model_conds"]["latent_shapes"].cond
                      for group in conds if group for c in group)
        parts = comfy.utils.unpack_latents(x_in, shapes)
        spans = self.plan.spans(shapes)
        accum = [[torch.zeros_like(p) for p in parts] if c is not None else None for c in conds]
        counts = [p.new_zeros([1, 1, p.shape[2], 1, 1] if i == 0 else [1, 1, 1, p.shape[3]])
                  for i, p in enumerate(parts)]
        for index, span in enumerate(spans):
            comfy.model_management.throw_exception_if_processing_interrupted()
            sub_x, sub_shapes = comfy.utils.pack_latents(slice_av(parts, span))
            sub_conds = [self.get_resized_cond(c, x_in, (index, span, sub_shapes)) for c in conds]
            outputs = calc_cond_batch(model, sub_conds, sub_x, timestep, window_options(model_options, span))
            split_outputs = [comfy.utils.unpack_latents(o, sub_shapes) for o in outputs]
            for modality, dim in ((0, 2), (1, 3)):
                start, end = span[:2] if modality == 0 else span[2:]
                length = end - start
                weights = torch.tensor(create_weights_pyramid(length), device=x_in.device, dtype=x_in.dtype)
                weight_shape = [1] * len(shapes[modality])
                weight_shape[dim] = length
                weights = weights.reshape(weight_shape)
                counts[modality].narrow(dim, start, length).add_(weights)
                for ci, group in enumerate(accum):
                    if group is not None:
                        group[modality].narrow(dim, start, length).add_(split_outputs[ci][modality] * weights)
        return [torch.zeros_like(x_in) if group is None else
                comfy.utils.pack_latents([p / count for p, count in zip(group, counts)])[0]
                for group in accum]
