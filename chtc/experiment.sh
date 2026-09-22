#!/bin/bash
# Job wrapper executed inside the container on a CHTC execute node.
#
# Every argument is forwarded verbatim to `script/exp_battery.py`, so the
# submit file decides which experiment runs:
#     arguments = sweep_ablations gemma-2-27b-it stimuli/general_eval.csv ...
#
# Caches (HF, Torch, Triton, TMPDIR) are pinned under the job scratch dir.
# Writing them to CHTC /home blows the home quota and slows every job.

set -euo pipefail

RUN_ROOT="$PWD"
LOG_FILE="$RUN_ROOT/experiment.log"
exec > >(tee -i "$LOG_FILE") 2>&1

echo "host=$(hostname -f) cluster=${CLUSTER_ID:-NA} process=${PROCESS_ID:-NA}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"

# ---- caches in scratch, never in /home ----
export HF_HOME="$RUN_ROOT/cache/hf"
export HF_DATASETS_CACHE="$RUN_ROOT/cache/hf_datasets"
export TORCH_HOME="$RUN_ROOT/cache/torch"
export TRITON_CACHE_DIR="$RUN_ROOT/cache/triton"
export TMPDIR="$RUN_ROOT/tmp"
export CACHE_DIR="$HF_HOME"
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TMPDIR"

# ---- unpack the code bundle ----
tar -xf llm-world-models.tar
cd llm-world-models
export PYTHONPATH="$PWD:${PYTHONPATH:-}"

# ---- credentials ----
if [ -n "${HF_TOKEN:-}" ]; then
    echo "Logging into Hugging Face..."
    hf auth login --token "$HF_TOKEN" 2>/dev/null \
        || huggingface-cli login --token "$HF_TOKEN"
fi

# ---- GPU utilization log ----
# Create the file first: transfer_output_files fails the job if an expected
# output is missing, and nvidia-smi is not always on PATH in the container.
GPU_METRICS="$RUN_ROOT/gpu_metrics.csv"
: > "$GPU_METRICS"
"$RUN_ROOT/gpu_monitor.sh" "$GPU_METRICS" 5 &
MONITOR_PID=$!
trap 'kill "$MONITOR_PID" 2>/dev/null || true; wait "$MONITOR_PID" 2>/dev/null || true' EXIT

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# ---- run ----
# SMOKE mode: `arguments = SMOKE <model_key> [timeout]` runs all three pipelines
# (inference / attribution ablation / attribution patching) over the 3-prompt
# slice via tools/smoke_test.sh, emitting RESULT| lines into this job's log.
if [ "${1:-}" = "SMOKE" ]; then
    shift
    MODEL="$1"; SMOKE_TO="${2:-1800}"
    echo "Smoke-testing $MODEL (per-method cap ${SMOKE_TO}s)"
    bash tools/smoke_test.sh "$MODEL" "$SMOKE_TO"
else
    echo "Running: $*"
    python -m script.exp_battery "$@"
fi

# ---- collect results ----
# Results are written under data/; hand them back through a single directory
# so transfer_output_files stays simple.
mkdir -p "$RUN_ROOT/result"
cp -r data/model/. "$RUN_ROOT/result/" 2>/dev/null || true
cp "$LOG_FILE" "$RUN_ROOT/result/" 2>/dev/null || true

echo "Job completed successfully."
