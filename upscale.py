"""Spatial upscaling at the native hybrid sampler's sequential/joint boundary."""

import copy
import hashlib
import logging
import os
from pathlib import Path
import pickle
import tempfile
import math

import torch
import torch.nn.functional as F

import comfy.utils
import comfy.sample
import folder_paths
import nodes
from comfy.nested_tensor import NestedTensor
from comfy.patcher_extension import WrappersMP
from comfy_api.latest import io

from .nodes import HybridStash, JointStage, add_outermost_wrapper, latent_signature, signatures_match
from .windows import WindowPlan, frame_at


def resize_video(video, size, mode="nearest-exact"):
    if tuple(video.shape[-2:]) == tuple(size):
        return video.clone()
    batch, channels, frames, height, width = video.shape
    flat = video.permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, height, width)
    flat = F.interpolate(flat, size=size, mode=mode)
    return flat.reshape(batch, frames, channels, *size).permute(0, 2, 1, 3, 4).contiguous()


def target_size(video, megapixels, align=32):
    height, width = video.shape[-2:]
    # Match the companion upscaler's 1024 squared pixels per MP and alignment.
    target_h = math.sqrt(megapixels * 1024 * 1024 * height / width)
    target_w = target_h * width / height
    return tuple(max(1, round(round(v / align) * align / 16)) for v in (target_h, target_w))


def joint_run(model):
    wrappers = model.get_wrappers(WrappersMP.OUTER_SAMPLE, "hybrid_joint")
    if len(wrappers) != 1 or not isinstance(wrappers[0], JointStage):
        raise ValueError("Connect the joint_model output of H3 Hybrid Windows to the upscaler.")
    run = wrappers[0].run
    if run is None or not run.active or run.stash is None or run.source is None:
        raise ValueError("Run the sequential sampler first, using H3 Hybrid Windows' latent output.")
    return run


def _converted_context(vae, video, width, height):
    layout = tuple((name, getattr(vae.first_stage_model, name, None))
                   for name in ("tiling", "tile_size", "tile_overlap_min", "clip_length", "token_drop"))
    identity = ("h3-warmup-v1", layout, str(vae.device), str(vae.patcher.patches_uuid), str(vae.vae_dtype),
                str(vae.vae_output_dtype()), tuple(video.shape), str(video.dtype), width, height)
    digest = hashlib.sha256(repr(identity).encode())
    source = video.detach().cpu().contiguous().view(torch.uint8).numpy()
    digest.update(memoryview(source).cast('B'))
    directory = Path(folder_paths.get_temp_directory()) / 'h3_warmup_cache'
    path = directory / (digest.hexdigest() + '.pt')
    if path.is_file():
        try:
            encoded = torch.load(path, map_location='cpu', weights_only=True)
        except (OSError, RuntimeError, EOFError, ValueError, pickle.UnpicklingError):
            logging.warning('[H3HybridWarmupPrior] Rebuilding unreadable warmup cache.')
        else:
            os.utime(path, None)
            return encoded.to(vae.output_device), True
    pixels = nodes.VAEDecode().decode(vae, {'samples': video})[0]
    pixels = comfy.utils.common_upscale(pixels.movedim(-1, 1), width, height, 'area', 'disabled').movedim(1, -1)
    encoded = vae.encode(pixels)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=directory, suffix='.tmp', delete=False) as f:
        temporary = f.name
    try:
        torch.save(encoded.detach().cpu(), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    entries = sorted(directory.glob('*.pt'), key=lambda p: p.stat().st_mtime_ns, reverse=True)
    for old in entries[32:]:
        old.unlink()
    return encoded, False


class H3HybridWarmupPrior(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3HybridWarmupPrior", display_name="H3 Resume Master for Low-Res Warmup",
            category="sampling/hybrid",
            description="Restore the selected native low-resolution video context saved with the finished master, using finished master audio. Older masters use the video VAE to bootstrap warmup. Keep the original master wired to the upscaler and final merge.",
            inputs=[io.Latent.Input("prior"), io.Int.Input("width", force_input=True),
                    io.Int.Input("height", force_input=True),
                    io.Vae.Input("vae", optional=True,
                                 tooltip="Bootstrap older masters or changed warmup dimensions by decoding, resizing pixels and re-encoding. Saved matching native context bypasses this conversion."),
                    io.Int.Input("completed_chunks", default=0, min=0, optional=True, force_input=True,
                                 tooltip="Connect RESUME C to resize just its previous context window. Zero preserves whole-master behavior for existing graphs."),
                    io.Int.Input("chunk_frames", default=243, min=5, optional=True, force_input=True),
                    io.Int.Input("overlap_frames", default=39, min=5, optional=True, force_input=True)],
            outputs=[io.Latent.Output("prior"), io.Int.Output("prior_start_chunk"), io.String.Output("report")])

    @classmethod
    def execute(cls, prior, width, height, vae=None, completed_chunks=0,
                chunk_frames=243, overlap_frames=39):
        if prior is None:
            return io.NodeOutput(None, 0, "Fresh run: no saved context to resize.")
        video, audio = prior["samples"].unbind()
        meta = prior.get("h3_meta", {})
        start_chunk = int(meta.get("start_chunk", 0))
        start, end = 0, video.shape[2]
        if completed_chunks:
            for key, value in (("chunk_frames", chunk_frames), ("overlap_frames", overlap_frames)):
                if value < 5 or (value - 5) % 17:
                    raise ValueError(f"{key} must use the H3 5+17k frame grid to preserve the resume timeline.")
                if meta.get(key) and meta[key] != value:
                    raise ValueError(f"Saved master has {key}={meta[key]}; restore that setting before resuming.")
            plan = WindowPlan(chunk_frames, overlap_frames, 1)
            start = (completed_chunks - 1 - start_chunk) * plan.stride
            end = start + plan.length
            if start < 0 or end > video.shape[2]:
                raise ValueError("The saved master does not contain the complete requested resume context window.")
            start_chunk = completed_chunks - 1
        a0 = round(frame_at(start) * 40 / 24)
        a1 = min(audio.shape[-1], round(frame_at(end) * 40 / 24))
        size = (height // 16, width // 16)
        decode_start, decode_end = start, end
        reused = False
        native = prior.get("h3_warmup")
        native_video = None
        fallback = "No native warmup was saved."
        if native is not None:
            nv = native["video"]
            grid = (native["chunk_frames"], native["overlap_frames"])
            expected = (meta.get("chunk_frames", chunk_frames), meta.get("overlap_frames", overlap_frames))
            if grid != expected:
                fallback = "Saved native warmup uses a different chunk grid."
            elif nv.ndim != 5 or nv.shape[:2] != video.shape[:2] or tuple(nv.shape[-2:]) != size:
                fallback = "Saved native warmup dimensions do not match this run."
            else:
                native_plan = WindowPlan(*grid, 1)
                offset = (native["start_chunk"] - int(meta.get("start_chunk", 0))) * native_plan.stride
                n0, n1 = start - offset, end - offset
                if n0 < 0 or n1 > nv.shape[2]:
                    fallback = "The requested context is outside the saved native warmup span."
                else:
                    native_video = nv[:, :, n0:n1].to(video).clone()
        if native_video is not None:
            working_video = native_video
        elif vae is not None and tuple(video.shape[-2:]) != size:
            # One neighboring 17-frame VAE group on each side retains the decoder's
            # blend and the encoder's final group when redoing an earlier chunk.
            decode_start = max(0, start - 5) if completed_chunks else start
            decode_end = min(video.shape[2], end + 5) if completed_chunks else end
            encoded, reused = _converted_context(vae, video[:, :, decode_start:decode_end], width, height)
            if encoded.shape != (*video.shape[:2], decode_end - decode_start, *size):
                raise ValueError("The video VAE must preserve the saved master's temporal layout when resizing the warmup copy.")
            working_video = (encoded[:, :, start - decode_start:end - decode_start].clone()
                             if completed_chunks else encoded)
        else:
            if native is not None and tuple(video.shape[-2:]) != size:
                raise ValueError(f"{fallback} Connect the video VAE to bootstrap this context.")
            working_video = resize_video(video[:, :, start:end], size, "area")
        working_audio = audio[..., a0:a1].clone() if completed_chunks else audio
        output = {**prior, "samples": NestedTensor([working_video, working_audio])}
        output.pop("h3_warmup", None)
        if completed_chunks:
            output["h3_meta"] = {**meta, "start_chunk": start_chunk}
        if prior.get("noise_mask") is not None:
            vm, am = prior["noise_mask"].unbind()
            if completed_chunks:
                vm = vm[:, :, start:end] if vm.shape[2] > 1 else vm
                am = am[..., a0:a1] if am.shape[-1] > 1 else am
            output["noise_mask"] = NestedTensor([resize_video(vm, size), am.clone()])
        scope = f"Context chunk {start_chunk + 1}" if completed_chunks else "Whole saved master"
        report = f"{scope}: {frame_at(end - start)} frames at {width}x{height}. "
        if native_video is not None:
            report += "Native low-resolution video restored with finished master audio; no VAE conversion. "
        else:
            report += (f"{fallback} Resize input: {frame_at(decode_end - decode_start)} "
                       f"of {frame_at(video.shape[2])} saved frames. ")
            if vae is not None and tuple(video.shape[-2:]) != size:
                report += "VAE bootstrap reused from cache. " if reused else "VAE bootstrap cached for retries. "
        report += "The original high-resolution master is retained separately."
        return io.NodeOutput(output, start_chunk, report)


class H3HybridLatentUpscale(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3HybridLatentUpscale", display_name="H3 Hybrid Latent Upscale - Target Megapixels",
            category="sampling/hybrid", is_experimental=True,
            description="Learned video latent upscale between sequential warmup and joint finishing. Keeps the handoff sigma, audio, masks and original accepted prefix. Joint sampling must use this node's model and latent with Disable Noise.",
            inputs=[io.Model.Input("joint_model"), io.Latent.Input("warmup"),
                    io.Combo.Input("model_name", options=folder_paths.get_filename_list("latent_upscale_models")),
                    io.Float.Input("megapixels", default=2.4, min=0.01, max=16., step=0.05,
                                   tooltip="Target area uses the companion upscaler's 1024 squared pixels per MP. Aspect ratio follows warmup; dimensions align to 32 pixels. Keep this setting fixed when resuming."),
                    io.Combo.Input("precision", options=["fp16", "bf16", "fp32"], default="fp16"),
                    io.Boolean.Input("enable_temporal_chunking", default=True),
                    io.Boolean.Input("keep_upscaler_loaded", default=False,
                                     tooltip="Keep the learned upscaler resident in VRAM instead of offloading it to CPU after every pass. Saves the host-to-device reupload on each call and between runs; leave it off when VRAM is tight."),
                    io.Latent.Input("prior", optional=True, lazy=True,
                                    tooltip="Original high-resolution saved master, before the low-resolution warmup copy. Required when resuming accepted frames.")],
            outputs=[io.Model.Output("joint_model"), io.Latent.Output("latent"), io.String.Output("report")])

    @classmethod
    def check_lazy_status(cls, joint_model, prior=None, **kwargs):
        return ["prior"] if joint_run(joint_model).accepted_v and prior is None else []

    @classmethod
    def execute(cls, joint_model, warmup, model_name, megapixels=2.4, precision="fp16",
                enable_temporal_chunking=True, keep_upscaler_loaded=False, prior=None):
        return cls._execute(joint_model, warmup, model_name, megapixels, precision,
                            enable_temporal_chunking, keep_upscaler_loaded, prior)

    @classmethod
    def _execute(cls, joint_model, warmup, model_name, megapixels, precision,
                 enable_temporal_chunking, keep_upscaler_loaded, prior, denoised=None):
        original = joint_run(joint_model)
        stash = original.stash
        base = joint_model.model
        packed, shapes = comfy.utils.pack_latents(warmup["samples"].unbind())
        if not signatures_match(stash.signature, latent_signature(packed, shapes)):
            raise ValueError("Connect output slot 0 of this chain's sequential sampler to warmup.")
        if not 0 < stash.sigma_end < 1:
            raise ValueError("Leave at least one joint finishing step after sequential warmup.")
        video, audio = warmup["samples"].unbind()
        size = target_size(video, megapixels)
        prior_video = prior_audio = None
        if original.accepted_v:
            if prior is None:
                raise ValueError("Connect the original saved master to prior to preserve accepted high-resolution frames.")
            pv, pa = prior["samples"].unbind()
            if tuple(pv.shape[-2:]) != size:
                raise ValueError(f"Saved master is {pv.shape[-1] * 16}x{pv.shape[-2] * 16}; target is {size[1] * 16}x{size[0] * 16}. Restore its target megapixels or start a fresh run.")
            start = (original.start_window - prior.get("h3_meta", {}).get("start_chunk", 0)) * original.plan.stride
            groups, remainder = divmod(start, 5)
            audio_start = round((17 * groups + sum((1, 4, 4, 4, 4)[:remainder])) * 40 / 24)
            if start < 0 or start + original.accepted_v > pv.shape[2] or audio_start + original.accepted_a > pa.shape[-1]:
                raise ValueError("The saved master does not contain this run's complete accepted window and audio.")
            prior_video = pv[:, :, start:start + original.accepted_v].clone()
            prior_audio = pa[..., audio_start:audio_start + original.accepted_a].clone()
        if size == tuple(video.shape[-2:]):
            return io.NodeOutput(joint_model, warmup, "Target matches warmup; the original hybrid handoff is unchanged.")

        upscaler = nodes.NODE_CLASS_MAPPINGS.get("MinimaxH3LatentUpscaler3D")
        if upscaler is None:
            raise RuntimeError("Install Comfyui_Minimax_h3_latent_Upscaler and restart ComfyUI.")

        def upscale(value):
            return upscaler.execute(
                {"samples": value}, model_name, {"mode": "megapixels", "megapixels": megapixels},
                32, enable_temporal_chunking, not keep_upscaler_loaded,
                "cuda" if torch.cuda.is_available() else "cpu", precision)[0]["samples"].float()

        high_noise = None
        if denoised is None:
            up_video = upscale(video)
        else:
            clean_video = denoised["samples"].unbind()[0]
            if clean_video.shape != video.shape:
                raise ValueError("Connect denoised_output from the same warmup sampler; its video shape must match warmup.")
            if original.all_accepted:
                up_video = resize_video(clean_video, size)
            else:
                up_video = upscale(clean_video)
                # Lift only the clean estimate. Nearest-resized residual noise
                # remains spatially correlated and corrupts the high-res finish.
                high_noise = comfy.sample.prepare_noise(
                    up_video, (stash.seed + 1) % (1 << 64), warmup.get("batch_index")).to(up_video)
                sampling = joint_model.get_model_object("model_sampling")
                sigma = up_video.new_tensor(stash.sigma_end)
                raw_video = sampling.noise_scaling(
                    sigma, high_noise, base.latent_format.process_in(up_video))
                up_video = base.latent_format.process_out(sampling.inverse_noise_scaling(sigma, raw_video))
        source_video, source_audio = original.source
        vm = warmup.get("noise_mask")
        vm = vm.unbind()[0] if vm is not None else None
        has_source_pins = vm is not None and torch.any(vm[:, :, original.accepted_v:] < 1)
        source_video = upscale(source_video) if has_source_pins else resize_video(source_video, size, "bilinear")
        source_audio = source_audio.clone()
        if prior_video is not None:
            source_video[:, :, :original.accepted_v] = prior_video.to(source_video)
            source_audio[..., :original.accepted_a] = prior_audio.to(source_audio)
        output = {**warmup, "samples": NestedTensor([up_video, audio.clone()])}
        if warmup.get("noise_mask") is not None:
            mask_video, mask_audio = warmup["noise_mask"].unbind()
            output["noise_mask"] = NestedTensor([resize_video(mask_video, size), mask_audio.clone()])
        high_packed, high_shapes = comfy.utils.pack_latents(output["samples"].unbind())
        run = copy.copy(original)
        run.source = [source_video, source_audio]
        if original.all_accepted:
            output["samples"] = NestedTensor([source_video, source_audio])
            high_packed, high_shapes = comfy.utils.pack_latents(run.source)
            raw = noise = None
        else:
            _, raw_audio = comfy.utils.unpack_latents(stash.raw_state, shapes)
            noise_video, noise_audio = comfy.utils.unpack_latents(stash.noise, shapes)
            sampling = joint_model.get_model_object("model_sampling")
            model_video = base.latent_format.process_in(up_video)
            raw_video = sampling.noise_scaling(torch.tensor(stash.sigma_end), torch.zeros_like(model_video), model_video)
            raw, _ = comfy.utils.pack_latents([raw_video, raw_audio])
            video_noise = resize_video(noise_video, size) if high_noise is None else high_noise
            noise, _ = comfy.utils.pack_latents([video_noise, noise_audio])
        run.stash = HybridStash(latent_signature(high_packed, high_shapes), stash.sigma_end, stash.seed, raw, noise)
        patched = joint_model.clone()
        add_outermost_wrapper(patched, WrappersMP.OUTER_SAMPLE, "hybrid_joint", JointStage(run))
        noise_report = ("Audio and the remaining schedule are retained; no fresh noise is added. "
                        if denoised is None else
                        f"Denoised video upscaled; target-resolution video noise seed {(stash.seed + 1) % (1 << 64)}. "
                        "Captured audio state and the remaining schedule are retained. ")
        quality_report = ("Learned upscaling of a partial latent is experimental; use the joint output for decoding and saving."
                          if denoised is None else
                          "Use this model and latent with Disable Noise for joint finishing; keep target MP fixed on resume.")
        report = (f"Hybrid handoff: {video.shape[-1] * 16}x{video.shape[-2] * 16} -> {size[1] * 16}x{size[0] * 16} "
                  f"({megapixels:g} target MP), sigma {stash.sigma_end:.9f}.\n"
                  f"{noise_report}Accepted video rows restored from the original master: {original.accepted_v}.\n"
                  f"{quality_report}")
        return io.NodeOutput(patched, output, report)


class H3HybridCleanLatentUpscale(H3HybridLatentUpscale):
    @classmethod
    def define_schema(cls):
        schema = super().define_schema()
        schema.node_id = "H3HybridCleanLatentUpscale"
        schema.display_name = "H3 Hybrid Clean Latent Upscale - Target Megapixels"
        schema.description = ("Upscale the warmup's denoised video estimate and rebuild video noise at the captured "
                              "handoff sigma. Retains audio and the original accepted prefix for native joint finishing.")
        schema.inputs.insert(2, io.Latent.Input("denoised", tooltip="Connect denoised_output (slot 1) of the same warmup sampler."))
        return schema

    @classmethod
    def execute(cls, joint_model, warmup, denoised, model_name, megapixels=2.4, precision="fp16",
                enable_temporal_chunking=True, keep_upscaler_loaded=False, prior=None):
        return cls._execute(joint_model, warmup, model_name, megapixels, precision,
                            enable_temporal_chunking, keep_upscaler_loaded, prior, denoised)
