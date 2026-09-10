#!/bin/bash
# Same colour-only grade on both renders (the recipe approved on video_00024 / 00028): chroma tint+saturation only,
# source-trajectory targets from stream_00170, 1-8 s reference protected, 2 s smoothing/ramp, audio copied.
set -e
SRC=/workspace/ComfyUI/input/stream_00170.mp4
QA=/workspace/analysis/hybrid_vs_looping_stream00170_2026-09-10
for f in "$@"; do
  stem="${f%.mp4}"
  python3 /workspace/scripts/h3_finish_grade.py "$f" --out "${stem}_color.mp4" --qa-dir "$QA/color_qa_$(basename "$stem")" \
    --source "$SRC" --mask none --chunk-frames 243 --overlap-frames 39 --ref-start-sec 1 --ref-end-sec 8 \
    --luma 0 --max-gamma 1 --contrast 0 --chroma 0.85 --allow-boost --gain-floor 0.9 --sharp 0 --mid-band 0 \
    --denoise 0 --demottle 0 --knee 0 --audio copy --smooth-seconds 2 --ramp-seconds 2 --max-chroma-shift 0.025 \
    --threads 8 --workers 4 --crf 12
done
