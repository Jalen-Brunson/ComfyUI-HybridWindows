# Keyframes at frame numbers

**H3 Hybrid Keyframes** (`H3HybridKeyframes`) pins stills at frame numbers of the
**whole run** on the native two-sampler chain. Plug each image into its own socket
(a new `keyframe` socket appears as you connect one), list the frame numbers in the
same order, and connect the output to the `keyframes` input of **H3 Hybrid Windows**.

```
Load Image ─┐
Load Image ─┼─ H3 Hybrid Keyframes ── keyframes ──▶ H3 Hybrid Windows
Load Image ─┘   frame_indices: 0, 15, 32, 64, 100
```

Nothing changes on the window encoders or the samplers. Two examples wire it:
`H3 Hybrid FL2VA Keyframes - native KSampler Advanced.json` pins the bundled first and
last stills at `0, -1` with the encoders' own first/last inputs empty, and
`H3 Hybrid R2V Keyframes - native KSampler Advanced.json` pins the reference portrait at
`0, -1` beside its `<Picture 1>` role, on a portrait canvas so the photo is not cropped.

## What Hybrid Windows does with them

Core's **Add Guide for MiniMax H3** anchors one image per node at an index that counts
from the start of that conditioning's own window. Hybrid Windows takes the global
number instead and, for each still:

1. resolves negatives (`-1` is the last frame of the run);
2. finds the window(s) whose frames contain it. A frame inside a shared overlap is given
   to **both** windows, each at its own index, since the joint stage fuses both there;
3. converts the number to that window's own index;
4. fits the still to the canvas and encodes it with the VAE once;
5. appends the same `minimax_keyframes` entry core's guide node writes. If the encoder
   already carries a guide at that index (an FL2VA `first_frame` at 0, say), the still
   replaces it, and the report says so.

Both stages then scope the entry per window exactly as they scope a core guide. The
CPU test `tests/test_keyframes.py` runs the chain on the pack's stand-in model and
checks that every window's model calls, in both stages, see only their own guides at
the converted indices.

With 243-frame windows and a 39-frame overlap:

| Window | Frames of the run | Frame 300 becomes |
|---|---:|---|
| 1 | 0 – 242 | – |
| 2 | 204 – 446 | window 2, frame 96 |
| 3 | 408 – 650 | – |

The Hybrid Windows **report** lists every placement, for example:

```
Keyframes: 3 still(s) on the 480x832 canvas (size from set on H3 Hybrid Keyframes); frame numbers count over this run's 651 frames.
  keyframe 1 (keyframe_0): frame 0 -> window 1 at its frame 0
  keyframe 2 (keyframe_1): frame 230 -> window 1 at its frame 230 and window 2 at its frame 26 (shared overlap)
  keyframe 3 (keyframe_2): frame 650 -> window 3 at its frame 242, fitted by crop from 1080x1920
```

## Inputs

| Input | Meaning |
|---|---|
| `vae` | The H3 video VAE the encoders use. |
| `frame_indices` | One number per image, socket order, separated by commas: `0, 15, 32, 64, 100`. Negative numbers count from the end. A socket carrying a batch uses one number per frame of the batch. |
| `keyframe_N` | The stills. Any size; each is fitted to the canvas. |
| `fit` | `crop` keeps the aspect and trims the edges (core's behaviour); `stretch` keeps every pixel and distorts. |
| `width`, `height` | The canvas. At `0` Hybrid Windows takes the size from the master latent connected to it, else from the encoders' own guides. With plain prompts and no master latent, set both to the encoders' width and height (the example connects the size nodes). |

The node's own `report` restates what it received; the Hybrid Windows report shows
where each frame landed. Errors name the problem before sampling: a count mismatch
between images and numbers, a frame outside the run, a frame listed twice, a canvas
nobody can determine.

## Limits

- Native chain only. The single-node **H3 Hybrid Window Sampler** still refuses
  conditioning that carries keyframe guides.
- A keyframe is conditioning, not pasted pixels: the model is guided toward the still
  at that moment. **FL2VA** is the checkpoint trained on frame anchors; on **Ref2VA**
  guides coexist with references, and mid-video anchors there are untested.
- Frame numbers count over the latent being sampled in **this run**. On an
  accepted-prefix resume the run's latent is the continuation master; a still inside the
  accepted prefix is reported and changes nothing.
- A still shared by two windows is encoded once, so memory cost is one latent frame
  per keyframe.
