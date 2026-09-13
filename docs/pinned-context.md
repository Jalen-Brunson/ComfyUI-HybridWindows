# Pinned video context

Both **H3 Hybrid Windows** (`H3HybridWindows`, native two-sampler chain) and
**H3 Hybrid Window Sampler** (`MMH3HybridWindowSampler`, original single-node 05
sampler) support `overlap_pin = tail_custom` with `context_frames = 5`.

The bundled native Ref2VA/FL2VA and single-sampler Ref2VA examples select five
frames. Existing nodes and API calls that omit the new options retain `full`
pinning. The 05N workflow builder also selects five frames. The accessible
05A inpaint workflow keeps full pinning so its optional source mask remains
supported.

## Overlap versus context

**Overlap** is the shared timeline region between adjacent windows. **Pinned
context** is how much of that region stays anchored to the preceding generated
result during sequential warmup.

For 243-frame windows and a 39-frame overlap:

| Region in the next window | Frames | Warmup behavior with context 5 |
|---|---:|---|
| Earlier overlap | 34 | May regenerate internally |
| Final overlap next to the new frames | 5 | Anchored to the preceding result |
| New output | 204 | Generated normally |

All 39 overlap frames still reach the model. Releasing pins does not remove
history, trim the overlap or change chunk duration. Temporary overlap changes
do not replace its previously owned prediction or rewrite accepted output.
Audio carry, source-control alignment, reference conditioning and the joint
finish retain their ordinary behavior. Generated fresh audio can still respond
to changed video conditioning.

## Controls

- `overlap_pin = full`: pin the complete video overlap, as before.
- `overlap_pin = tail_custom`: use the `context_frames` value.
- `context_frames = 5`: the five-frame setting used in the continuation tests.
- `context_frames = 0`: release all temporary video pins. Overlap, references,
  audio and joint-stage context remain, so this is not an independent chunk.

H3 represents groups of video frames with latent rows. Context rounds **up** to
the smallest whole-row suffix covering the requested frames, capped by the
actual incoming overlap. With a regular 39-frame overlap, useful exact values
include **5, 9, 13, 17, 22 and 39**; **6 rounds up to 9**. The sampler report shows
the effective count. A shortened final window can have a larger overlap, so use
`full` when the intent is to pin all of it regardless of its length.

Partial pinning currently requires full-frame video regeneration outside any
accepted prefix. An inpaint mask protecting fresh source video requires `full`.
Reference images, RefMods and source/control video conditioning are compatible
with partial pinning when they do not impose such a preservation mask.

## Validation and limits

CPU tests exercise the native ComfyUI sampling path with a small H3 stand-in:
zero, five, rounded, longer and capped context; exact accepted video/audio
preservation; multiple fresh windows; resume offsets; shortened final windows;
and fully sequential or joint finishing. The single-node and native paths are
compared using the same noise draws and masks.

Five-frame pinning produced a promising six-chunk production result with less
observed buildup. That comparison also changed other settings, so it does not
establish a general mottling fix or equivalence to independent no-context chunks.
