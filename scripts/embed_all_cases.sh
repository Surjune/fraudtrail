#!/usr/bin/env bash
# Embed the closed-case notes in slices, one process each.
# A long ONNX run dies on this machine without an exception; a fresh process per slice
# keeps a failure to one slice, and the slice can be repeated by its offset.
set -u
PY=.venv/Scripts/python.exe
SLICE=${SLICE:-800}
BATCH=${BATCH:-300}
TOTAL=${TOTAL:-5565}
for (( offset=0; offset<TOTAL; offset+=SLICE )); do
  for attempt in 1 2; do
    echo "=== offset $offset (attempt $attempt) ==="
    if "$PY" scripts/build_embeddings.py --only cases --batch "$BATCH" \
         --offset "$offset" --limit "$SLICE" 2>&1 | grep -viE "it/s\]|warn"; then
      break
    fi
    echo "!!! slice at offset $offset failed"
  done
done
echo "=== all slices attempted ==="
