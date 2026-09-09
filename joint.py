"""Hybrid joint-stage conditioning, context handling and mask preparation."""
import torch
from comfy.context_windows import get_matching_context_schedule, get_matching_fuse_method

from .dependencies import _mmh3_module

# Loaded on first sampling execution, after custom-node registration is complete.
common = _mmh3_module("common")
loop = _mmh3_module("nodes_looping_sampler")
frame_at_latent = common.frame_at_latent
unpack_av = common.unpack_av
per_row_mask_is_continuous = _mmh3_module("nodes_loop").per_row_mask_is_continuous
MMH3ContextHandler = _mmh3_module("nodes_windows").MMH3ContextHandler
_audio_profile_to_mask = loop._audio_profile_to_mask
_mask_to_video_latent = loop._mask_to_video_latent
_sliced_mask = loop._sliced_mask
_split_mask = loop._split_mask
_video_mask_to_audio = loop._video_mask_to_audio

_WINDOW_KEY = "mmh3_joint_window_index"


def bind_window_conditioning(cond_set, windows):
    conds = cond_set["conds"]
    if len(conds) != len(windows):
        raise ValueError(
            f"Joint sampler planned {len(windows)} windows but received {len(conds)} "
            "conditioning sets. Match total, window and overlap frames in the "
            "reference encoder and sampler.")
    bound = []
    for i, cond in enumerate(conds):
        if not cond:
            raise ValueError(f"Window {i + 1} has no conditioning.")
        for tokens, metadata in cond:
            if metadata.get("minimax_keyframes"):
                raise ValueError("Joint sampler does not support keyframe guides yet.")
            bound.append([tokens, {**metadata, _WINDOW_KEY: i}])
    return bound


class MMH3JointContextHandler(MMH3ContextHandler):
    def __init__(self, windows, length, overlap, accumulator_device="gpu", freenoise=False):
        super().__init__(
            context_schedule=get_matching_context_schedule("standard_static"),
            fuse_method=get_matching_fuse_method("pyramid"),
            context_length=length, context_overlap=overlap, context_stride=1,
            closed_loop=False, dim=2, freenoise=freenoise, causal_window_fix=False,
            split_conds_to_windows=False, accumulator_device=accumulator_device,
        )
        self.window_indices = {
            tuple(window.index_list): i for i, window in enumerate(windows)
        }

    def get_context_windows(self, model, x_in, model_options):
        windows = super().get_context_windows(model, x_in, model_options)
        if {tuple(w.index_list) for w in windows} != set(self.window_indices):
            raise ValueError("Sampling windows changed after conditioning was assigned.")
        return windows

    def get_resized_cond(self, cond_in, x_in, window, device=None):
        if cond_in is None:
            return None
        index = self.window_indices[tuple(window.index_list)]
        selected = [c for c in cond_in if c.get(_WINDOW_KEY) == index]
        if not selected:
            raise ValueError(f"Missing conditioning for joint window {index + 1}.")
        return super().get_resized_cond(selected, x_in, window, device)

    def evaluate_context_windows(self, calc_cond_batch, model, x_in, conds,
                                 timestep, enumerated_context_windows,
                                 model_options, window_state, total_windows=None,
                                 device=None, first_device=None):
        results = []
        for item in enumerated_context_windows:
            # Core writes context_window into these options. Keep each window's
            # control offset local, including on exceptions or cancelled renders.
            options = {**model_options, "transformer_options": {
                **model_options.get("transformer_options", {}),
                "mmh3_control_frame0": frame_at_latent(item[1].index_list[0]),
            }}
            results.extend(super().evaluate_context_windows(
                calc_cond_batch, model, x_in, conds, timestep, [item], options,
                window_state, total_windows, device, first_device))
        return results


def prepare_masks(latent, denoise_mask, audio_denoise_mask, mode):
    video, audio = unpack_av(latent, allow_video_only=True)
    total_t = video.shape[2]
    total_a = 0 if audio is None else audio.shape[3]
    mask_v, mask_a = _split_mask(latent)
    if denoise_mask is not None or audio_denoise_mask is not None:
        if not per_row_mask_is_continuous():
            raise RuntimeError("Joint sampler masks require H3 per-row mask support.")
    if denoise_mask is not None:
        mv = _mask_to_video_latent(
            denoise_mask.to(video.device), total_t, video.shape[3], video.shape[4], mode)
        mask_v = mv if mask_v is None else torch.minimum(mask_v.to(mv), mv)
    if audio_denoise_mask is not None:
        if audio is None:
            raise ValueError("An audio mask needs an AV latent containing audio.")
        am = _mask_to_video_latent(audio_denoise_mask.to(video.device), total_t, 1, 1, mode)
        profile = _video_mask_to_audio(am, total_t, total_a, mode)
        ma = _audio_profile_to_mask(profile, audio)
        mask_a = ma if mask_a is None else torch.minimum(mask_a.to(ma), ma)
    out = dict(latent)
    if mask_v is not None or mask_a is not None:
        out["noise_mask"] = _sliced_mask(
            video, audio, mask_v, mask_a, 0, total_t, 0, total_a)
    return out
