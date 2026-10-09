#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then echo "usage: $0 TOPDOWN_IMAGE [OUTPUT_DIR]" >&2; exit 2; fi
image=$1
output=${2:-outputs/topdown_run}
exec python -m pipeline.run_pipeline --input-topdown "$image" --config configs/image_input_example.yaml --output "$output"
