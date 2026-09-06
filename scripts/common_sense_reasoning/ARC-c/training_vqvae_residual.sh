#!/usr/bin/env bash
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$REPO_ROOT/scripts/common_sense_reasoning/ARC-c/training_vqvae_refined.sh" \
    --reference-residual --output-dir "$REPO_ROOT/outputs/vqvae_train_arc_c_residual" \
    --steps 3000 --joint-steps 2000 --learning-rate 0.0001 "$@"
