"""Aligned source guides encoded at the sampling stage's actual resolution."""

import logging

import comfy.utils
from comfy.nested_tensor import NestedTensor
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import MiniMaxH3AddGuide

from .windows import WINDOW_KEY, frame_at, slice_av


def guide_inputs():
    return [
        io.Image.Input("guide_video", optional=True,
                       tooltip="Aligned source frames at 24 fps, already trimmed to this run including resume context. Encoded per window at each sampling stage's actual resolution."),
        io.Vae.Input("guide_vae", optional=True,
                     tooltip="H3 video VAE, required when guide_video is connected."),
        io.Audio.Input("guide_audio", optional=True,
                       tooltip="Optional aligned soundtrack, already trimmed to this run. Conditions the model; audio preservation still uses the denoise mask."),
        io.Vae.Input("guide_audio_vae", optional=True,
                     tooltip="H3 audio VAE, required when guide_audio is connected."),
    ]


def append_guide(metadata, guide):
    return {**metadata, "minimax_keyframes": list(metadata.get("minimax_keyframes", [])) + [guide]}


class AlignedGuides:
    def __init__(self, video=None, vae=None, audio=None, audio_vae=None):
        if video is not None and vae is None:
            raise ValueError("Connect guide_vae when using guide_video.")
        if audio is not None and audio_vae is None:
            raise ValueError("Connect guide_audio_vae when using guide_audio.")
        self.video = video
        self.vae = vae
        self.audio = audio
        self.audio_vae = audio_vae

    def encode(self, parts, spans):
        last_frame = frame_at(max(span[1] for span in spans))
        if self.video is not None and len(self.video) < last_frame:
            raise ValueError(f"Hybrid guides need {last_frame} source frames, but guide_video has "
                             f"{len(self.video)}. Include the current run's resume context.")
        if self.audio is not None:
            sr = self.audio["sample_rate"]
            if self.audio["waveform"].shape[-1] + round(sr / 24) < round(last_frame * sr / 24):
                raise ValueError("guide_audio is shorter than this run. Supply its aligned soundtrack or disconnect guide_audio.")
        guides = []
        for index, span in enumerate(spans):
            v0, v1, _, _ = span
            if v0 % 5 or (v1 - v0 - 2) % 5:
                raise ValueError("Hybrid guide windows must align to H3's causal 17-frame groups.")
            f0, f1 = frame_at(v0), frame_at(v1)
            audio = None
            if self.audio is not None:
                audio = {**self.audio, "waveform": self.audio["waveform"][..., round(f0 * sr / 24):round(f1 * sr / 24)]}
            latent = {"samples": NestedTensor(slice_av(parts, span))}
            guided = MiniMaxH3AddGuide.execute(
                positive=[[None, {}]], latent=latent, frame_idx=0, vae=self.vae,
                image=None if self.video is None else self.video[f0:f1],
                audio=audio, audio_vae=self.audio_vae).args[0]
            guide = guided[0][1]["minimax_keyframes"][-1]
            if self.video is not None and tuple(guide["latent"].shape[2:]) != (v1-v0, *parts[0].shape[3:]):
                raise ValueError("Encoded guide shape does not match its H3 sampling window.")
            guides.append(guide)
            logging.info("[Hybrid guides] window %d: run frames %d..%d -> local frame 0, %dx%d, audio %s",
                         index + 1, f0, f1 - 1, parts[0].shape[4] * 16, parts[0].shape[3] * 16,
                         "on" if audio is not None else "off")
        return guides


class GuideStage:
    """Prepare positive guides before model loading; discard encodings after sampling."""

    def __init__(self, plan, guides):
        self.plan = plan
        self.guides = guides

    def __call__(self, executor, noise, latent_image, sampler, sigmas, denoise_mask=None,
                 callback=None, disable_pbar=False, seed=None, latent_shapes=None):
        if len(sigmas) < 2:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, callback,
                            disable_pbar, seed, latent_shapes=latent_shapes)
        guider = executor.class_obj
        original = guider.conds
        parts = comfy.utils.unpack_latents(latent_image, latent_shapes)
        guides = self.guides.encode(parts, self.plan.spans(latent_shapes))
        positive = [append_guide(cond, guides[cond[WINDOW_KEY]]) for cond in original["positive"]]
        guider.conds = {**original, "positive": positive}
        try:
            return executor(noise, latent_image, sampler, sigmas, denoise_mask, callback,
                            disable_pbar, seed, latent_shapes=latent_shapes)
        finally:
            guider.conds = original
