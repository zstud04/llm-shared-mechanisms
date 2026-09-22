#!/bin/bash
# Sample nvidia-smi into a CSV so every job can answer "were the GPUs busy?".
# Writes a fallback row rather than failing when nvidia-smi is absent from the
# container PATH — a missing output file would hold the job at transfer time.

OUT="${1:-gpu_metrics.csv}"
INTERVAL="${2:-5}"

HEADER="timestamp,index,name,utilization_gpu_pct,utilization_mem_pct,memory_used_mb,memory_total_mb,power_w,temperature_c"
echo "$HEADER" > "$OUT"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "$(date -Is),NA,nvidia-smi-unavailable,NA,NA,NA,NA,NA,NA" >> "$OUT"
    exit 0
fi

while true; do
    nvidia-smi \
        --query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu \
        --format=csv,noheader,nounits >> "$OUT" 2>/dev/null || true
    sleep "$INTERVAL"
done
