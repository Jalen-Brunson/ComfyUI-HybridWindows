"""Build two matched API prompts for stream_00170: native HybridWindows (6+2) vs MMH3LoopingSampler (8 sequential).
Everything but the sampler is identical: same source decode + noise, same ref image, same model chain, dims, windows, seed, prompt."""
import json, sys
OUT = "/workspace/analysis/hybrid_vs_looping_stream00170_2026-09-10"
SRC_FILE = "stream_00170.mp4"          # ComfyUI/input
REF_IMAGE = "hybrid_man_reference.png" # ComfyUI/input
W, H = 768, 576                        # source native dims (both /32)
L, OV = 243, 39
N_FRAMES = 2895                        # = 243 + 13*204 -> 14 windows, whole source
N_WIN = 1 + (N_FRAMES - L) // (L - OV)
assert L + (N_WIN - 1) * (L - OV) == N_FRAMES, N_WIN
FPS = 24.0
SEED = 123
STEPS, SWITCH = 8, 6
PROMPT = (
    "Use <Video 1> as the continuous motion, timing and camera-framing reference. "
    "Replace the woman in <Video 1> with the man in <Picture 1>: a fully human man with shoulder-length light brown hair, "
    "a full brown beard and moustache, blue eyes, wearing a plain black shirt, with no robotic or metal parts anywhere on his face or body. "
    "He walks slowly toward the camera down the rubble-strewn street of a ruined, grey, bombed-out city under an overcast sky, "
    "looking into the lens and talking, matching the body movement, head motion, gestures, mouth movement and pacing of <Video 1> exactly. "
    "Keep his appearance, the dusty daylight, the setting and the framing consistent throughout this single continuous shot. "
    "Steady camera tracking backward in front of him. He speaks in a calm, natural male voice over quiet wind and distant ambience. "
    "No shot changes or zooms."
)

def common():
    p = {}
    # shared source chain (same ids in both prompts so the executor cache can reuse the decode)
    p["101"] = {"class_type": "LoadVideo", "inputs": {"file": SRC_FILE}}
    p["102"] = {"class_type": "Video Slice", "inputs": {"video": ["101", 0], "start_time": 0.0, "duration": N_FRAMES / FPS, "strict_duration": True}}
    p["103"] = {"class_type": "GetVideoComponents", "inputs": {"video": ["102", 0]}}
    p["104"] = {"class_type": "ImageAddNoise", "inputs": {"image": ["103", 0], "seed": SEED, "strength": 0.1}}
    p["1"] = {"class_type": "LoadImage", "inputs": {"image": REF_IMAGE}}
    p["2"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "minimax_h3_ref2va_int8_convrot.safetensors", "weight_dtype": "default"}}
    p["3"] = {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "minimax/MiniMax-H3-Ref2VA-Acc-8Step_comfy.safetensors", "strength_model": 1.0, "model": ["2", 0]}}
    p["4"] = {"class_type": "MiniMaxH3SigmaShift", "inputs": {"shift_video": 12.0, "shift_audio": 3.0, "model": ["3", 0]}}
    p["5"] = {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl_32b_minimax_h3_int8_convrot.safetensors", "type": "minimax", "device": "default"}}
    p["6"] = {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_video_vae_int8_convrot.safetensors"}}
    p["7"] = {"class_type": "VAELoader", "inputs": {"vae_name": "minimax_h3_audio_vae_fp32.safetensors"}}
    return p

def save_chain(p, samples_node, prefix):
    p["25"] = {"class_type": "VAEDecode", "inputs": {"samples": [samples_node, 0], "vae": ["6", 0]}}
    p["26"] = {"class_type": "VAEDecodeAudio", "inputs": {"samples": [samples_node, 0], "vae": ["7", 0]}}
    p["28"] = {"class_type": "CreateVideo", "inputs": {"fps": FPS, "bit_depth": 8, "color_space": "sRGB", "images": ["25", 0], "audio": ["26", 0]}}
    p["29"] = {"class_type": "SaveVideo", "inputs": {"filename_prefix": prefix, "format": "mp4", "format.codec": "h264",
                                                      "format.codec.encoding": "re-encode", "format.codec.encoding.crf": 16.0, "codec": "auto", "video": ["28", 0]}}

def hybrid_native():
    p = common()
    hw = {"class_type": "H3HybridWindows", "inputs": {"model": ["4", 0], "window_frames": L, "overlap_frames": OV, "total_steps": STEPS}}
    for i in range(N_WIN):
        sid, cid = str(200 + i), str(300 + i)
        p[sid] = {"class_type": "ImageFromBatch", "inputs": {"image": ["104", 0], "batch_index": i * (L - OV), "length": L}}
        p[cid] = {"class_type": "MiniMaxH3ReferenceToVideo", "inputs": {
            "prompt": PROMPT, "width": W, "height": H, "length": L, "ref_image_size": "match",
            "clip": ["5", 0], "vae": ["6", 0], "ref_images.ref_image_0": ["1", 0], "ref_videos.ref_video_0": [sid, 0]}}
        hw["inputs"][f"prompts.positive_{i}"] = [cid, 0]
    p["19"] = hw
    p["20"] = {"class_type": "EmptyMiniMaxH3LatentAV", "inputs": {"width": W, "height": H, "length": ["19", 3]}}
    p["21"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["19", 2]}}
    p["22"] = {"class_type": "KSamplerAdvanced", "inputs": {"add_noise": "enable", "noise_seed": SEED, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
               "start_at_step": 0, "end_at_step": SWITCH, "return_with_leftover_noise": "enable", "positive": ["19", 2], "negative": ["21", 0], "model": ["19", 0], "latent_image": ["20", 0]}}
    p["23"] = {"class_type": "KSamplerAdvanced", "inputs": {"add_noise": "disable", "noise_seed": SEED, "steps": STEPS, "cfg": 1.0, "sampler_name": "euler", "scheduler": "simple",
               "start_at_step": SWITCH, "end_at_step": STEPS, "return_with_leftover_noise": "disable", "positive": ["19", 2], "negative": ["21", 0], "model": ["19", 1], "latent_image": ["22", 0]}}
    save_chain(p, "23", "video/compare_stream00170/hybrid_native_6p2")
    return p

def mmh3_looping():
    p = common()
    p["40"] = {"class_type": "MMH3ReferenceMultiPrompt", "inputs": {
        "clip": ["5", 0], "vae": ["6", 0], "audio_vae": ["7", 0], "width": W, "height": H, "length": N_FRAMES,
        "ref_image_size": "match", "prompts": PROMPT, "use_input_audio": False, "unload_text_encoder": True,
        "ref_images": ["1", 0], "ref_videos.ref_video_0": ["104", 0],
        "window_ref_video": True, "chunk_frames": L, "overlap_frames": OV}}
    p["41"] = {"class_type": "MMH3CondSelect", "inputs": {"cond_set": ["40", 0], "index": 0}}
    p["42"] = {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["41", 0]}}
    p["43"] = {"class_type": "BasicGuider", "inputs": {"model": ["4", 0], "conditioning": ["42", 0]}}
    p["44"] = {"class_type": "RandomNoise", "inputs": {"noise_seed": SEED}}
    p["45"] = {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}}
    p["46"] = {"class_type": "BasicScheduler", "inputs": {"model": ["4", 0], "scheduler": "simple", "steps": STEPS, "denoise": 1.0}}
    p["47"] = {"class_type": "MMH3LoopingSampler", "inputs": {
        "noise": ["44", 0], "guider": ["43", 0], "sampler": ["45", 0], "sigmas": ["46", 0], "cond_set": ["40", 0], "latent": ["40", 1],
        "chunk_frames": L, "overlap_frames": OV, "carry": "mask", "overlap_strength_video": 1.0, "overlap_strength_audio": 0.9,
        "sampling_start_step": 0, "sampling_end_step": 1000, "phase2_start_step": 0, "keyframe_indices": "",
        "denoise_mask_mode": "max", "pin_renorm_video": 0.0, "carry_precomp": 0.0, "carry_noise_level": 0.0, "reroll_retries": 0}}
    save_chain(p, "47", "video/compare_stream00170/mmh3_looping_8seq")
    return p

if __name__ == "__main__":
    a, b = hybrid_native(), mmh3_looping()
    json.dump(a, open(f"{OUT}/hybrid_native.api.json", "w"), indent=1)
    json.dump(b, open(f"{OUT}/mmh3_looping.api.json", "w"), indent=1)
    print("windows", N_WIN, "frames", N_FRAMES, "seconds", N_FRAMES / FPS, "| native nodes", len(a), "| mmh3 nodes", len(b))
