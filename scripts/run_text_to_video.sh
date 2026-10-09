#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 1 ]]; then echo "usage: $0 SCENE_PROMPT [OUTPUT_DIR]" >&2; exit 2; fi
prompt=$1
output=${2:-outputs/text_run}
exec python -m pipeline.run_pipeline --scene-prompt "$prompt" --config configs/text_input_example.yaml --output "$output"
