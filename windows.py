"""H3 window geometry and prediction fusion for native ComfyUI sampling."""

from dataclasses import dataclass

import torch

import comfy.conds
import comfy.model_management
import comfy.utils
from comfy.context_windows import ContextHandlerABC, create_weights_pyramid
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy.nested_tensor import NestedTensor
from comfy_extras.nodes_minimax_h3 import align_frame_count, video_latent_t

WINDOW_KEY = "hybrid_window_index"
OFFSET_KEY = "hybrid_window_frame0"
# MMH3Tools' own key for the same quantity. Control-strength schedules and the
# Fun ControlNet wrapper read this one, so both stages publish both names and a
# graph built for either sampler behaves the same.
CONTROL_OFFSET_KEY = "mmh3_control_frame0"

AUDIO_PER_FRAME = 40 / 24


def frame_at(index):
    groups, remainder = divmod(index, len(FRAME_PER_TOKEN))
    return groups * sum(FRAME_PER_TOKEN) + sum(FRAME_PER_TOKEN[:remainder])


def audio_index_at(index, total_video, total_audio):
    """Audio latent column at a video latent boundary, clamped to the clip.

    The ends are pinned rather than computed so a window that reaches the last
    video latent also reaches the last audio column, whatever rounding would
    have said.
    """
    if index <= 0:
        return 0
    if index >= total_video:
        return total_audio
    return min(total_audio, max(0, round(frame_at(index) * AUDIO_PER_FRAME)))


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
        """Window spans for this latent, laid out like core's static schedule.

        A latent shorter than the rigid `length + (count-1)*stride` is accepted
        and its LAST window slides back to end at the clip, exactly as
        `create_windows_static_standard` does. A continuation master built by
        H3ContinueMaster is short in precisely this way on a project's last
        part, and refusing it would make resume impossible there.
        """
        video, audio = shapes
        total_v = video[2]
        rigid = video_latent_t(self.total_frames)
        smallest = 2 if self.count == 1 else (self.count - 2) * self.stride + self.length + 1
        if not smallest <= total_v <= rigid:
            raise ValueError(
                f"Hybrid windows: {self.count} conditioning window(s) need between {smallest} "
                f"and {rigid} video latents ({frame_at(smallest)}-{self.total_frames} frames); "
                f"this latent has {total_v} ({frame_at(total_v)} frames). Match the number of "
                "prompts to the latent, or connect total_frames to the Empty MiniMax H3 AV "
                "Latent length.")
        total_a = audio[3]
        if total_a != round(frame_at(total_v) * AUDIO_PER_FRAME):
            raise ValueError(
                f"Hybrid windows: {total_v} video latents pair with "
                f"{round(frame_at(total_v) * AUDIO_PER_FRAME)} audio latents, not {total_a}.")
        spans = []
        for index in range(self.count):
            v0 = index * self.stride
            v1 = v0 + self.length
            if v1 > total_v:  # core slides the final window back instead of shortening it
                v0, v1 = max(0, total_v - self.length), total_v
            spans.append((v0, v1, audio_index_at(v0, total_v, total_a),
                          audio_index_at(v1, total_v, total_a)))
        return spans


def plan_windows(window_frames, overlap_frames, count):
    return WindowPlan(align_frame_count(max(5, window_frames)),
                      align_frame_count(max(5, overlap_frames)), count)


def slice_av(parts, span):
    v0, v1, a0, a1 = span
    return [parts[0][:, :, v0:v1], parts[1][:, :, :, a0:a1]]


def ones_mask(video, audio):
    """A per-row keep-nothing mask shaped like MMH3's, so the two agree."""
    return [torch.ones([video.shape[0], 1] + list(video.shape[2:]),
                       dtype=torch.float32, device=video.device),
            torch.ones([audio.shape[0], 1, audio.shape[2], audio.shape[3]],
                       dtype=torch.float32, device=audio.device)]


def pin_prefix(mask, video, audio, accepted_v, accepted_a):
    """Hold the first accepted rows fixed, keeping any pin the master had.

    Keep-wins rather than overwrite: an inpaint keep region or a pinned source
    audio track inside the accepted span must stay pinned. At carry strength 1
    this is MMH3 `_carry_mask`'s clamp written out, so a graph pinned by either
    sampler sees the same mask.
    """
    if mask is None:
        vm, am = ones_mask(video, audio)
    elif isinstance(mask, NestedTensor):
        parts = mask.unbind()
        vm, am = parts[0].clone(), parts[-1].clone()
    else:  # a video-only mask; audio was never pinned
        vm, am = mask.clone(), ones_mask(video, audio)[1]
    vm[:, :, :accepted_v] = 0.0
    am[:, :, :, :accepted_a] = 0.0
    return NestedTensor([vm, am])


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
    frame0 = frame_at(span[0])
    return {**options, "transformer_options": {
        **options.get("transformer_options", {}),
        OFFSET_KEY: frame0, CONTROL_OFFSET_KEY: frame0,
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
