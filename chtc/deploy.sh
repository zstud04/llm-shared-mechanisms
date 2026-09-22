#!/bin/bash
# Bundle this repo, push it to a CHTC Access Point, and submit a job.
#
#   chtc/deploy.sh [options] -- <exp_battery args...>
#
#   -u USER      CHTC netid            (default: $CHTC_USER)
#   -s SERVER    Access Point          (default: ap2002.chtc.wisc.edu)
#   -f SUBFILE   submit file           (default: chtc/exp_run.sub)
#   -r RUN_ID    remote run directory  (default: run-<timestamp>)
#   -n           bundle and stage only; do not submit
#
# Examples:
#   chtc/deploy.sh -u mynetid -- general_eval gemma-2-9b-it \
#       stimuli/general_eval.csv general_instruct data/model/behavioral
#
#   chtc/deploy.sh -u mynetid -f chtc/exp_run_multigpu.sub -- \
#       sweep_ablations gemma-3-27b-it stimuli/general_eval.csv \
#       general_instruct data/model/ablation --method attribution
#
# CHTC requires Duo/2FA. This script opens one SSH ControlMaster and reuses it
# for every subsequent command, so you approve the push once per run.

set -euo pipefail

CHTC_USER="${CHTC_USER:-}"
CHTC_SERVER="ap2002.chtc.wisc.edu"
SUBFILE="chtc/exp_run.sub"
RUN_ID="run-$(date +%Y%m%d-%H%M%S)"
SUBMIT=1

while getopts "u:s:f:r:nh" opt; do
    case "$opt" in
        u) CHTC_USER="$OPTARG" ;;
        s) CHTC_SERVER="$OPTARG" ;;
        f) SUBFILE="$OPTARG" ;;
        r) RUN_ID="$OPTARG" ;;
        n) SUBMIT=0 ;;
        h) sed -n '2,25p' "$0"; exit 0 ;;
        *) exit 1 ;;
    esac
done
shift $((OPTIND - 1))
[ "${1:-}" = "--" ] && shift

if [ -z "$CHTC_USER" ]; then
    echo "Error: set -u USER or \$CHTC_USER" >&2
    exit 1
fi
if [ "$#" -lt 1 ] && [ "$SUBMIT" -eq 1 ]; then
    echo "Error: no exp_battery arguments given (use -n to stage only)" >&2
    exit 1
fi

EXP_ARGS="$*"
REMOTE_DIR="chtc-runs/$RUN_ID"
TARBALL="llm-world-models.tar"

# ---- shared SSH connection (one Duo prompt) ----
# CHTC_CONTROL_PATH lets an already-authenticated master socket (opened in a
# real terminal with a TTY for Duo) be reused from a non-interactive session.
if [ -n "${CHTC_CONTROL_PATH:-}" ]; then
    CONTROL_PATH="$CHTC_CONTROL_PATH"
    mkdir -p "$(dirname "$CONTROL_PATH")"
else
    CONTROL_DIR="$HOME/.ssh/controlmasters"
    mkdir -p "$CONTROL_DIR"
    chmod 700 "$CONTROL_DIR"
    CONTROL_PATH="$CONTROL_DIR/${CHTC_USER}-${CHTC_SERVER}.sock"
fi

if ! ssh -S "$CONTROL_PATH" -O check "$CHTC_USER@$CHTC_SERVER" 2>/dev/null; then
    echo "Opening SSH master to $CHTC_SERVER (approve Duo once)..."
    ssh -M -S "$CONTROL_PATH" -fNT \
        -o ControlPersist=8h -o ServerAliveInterval=60 -o ServerAliveCountMax=3 \
        "$CHTC_USER@$CHTC_SERVER"
fi
rsh() { ssh -S "$CONTROL_PATH" "$CHTC_USER@$CHTC_SERVER" "$@"; }

# ---- bundle the code (data and plots stay local) ----
echo "Bundling repo -> $TARBALL"
tar --exclude="./data" --exclude="./data_raw" --exclude="./analysis/plots" \
    --exclude="./.git" --exclude="*.tar" --exclude="__pycache__" \
    -cf "$TARBALL" \
    core utils script config stimuli tools environment.yml README.md

# ---- stage and submit ----
echo "Staging to $CHTC_SERVER:$REMOTE_DIR"
rsh "mkdir -p '$REMOTE_DIR/logs'"
scp -o ControlPath="$CONTROL_PATH" \
    "$TARBALL" "$SUBFILE" chtc/experiment.sh chtc/gpu_monitor.sh \
    "$CHTC_USER@$CHTC_SERVER:$REMOTE_DIR/"
rm -f "$TARBALL"

SUBFILE_NAME="$(basename "$SUBFILE")"

if [ "$SUBMIT" -eq 0 ]; then
    echo "Staged at $REMOTE_DIR (not submitted). Submit with:"
    echo "  ssh $CHTC_USER@$CHTC_SERVER 'cd $REMOTE_DIR && condor_submit $SUBFILE_NAME'"
    exit 0
fi

echo "Submitting: $EXP_ARGS"
# Pass HF_TOKEN through as a submit variable so gated repos (Llama, Gemma) can
# download inside the job. Submit files that don't reference $(HFTOKEN) ignore it.
rsh "cd '$REMOTE_DIR' && chmod +x experiment.sh gpu_monitor.sh && \
     condor_submit '$SUBFILE_NAME' EXP_ARGS='$EXP_ARGS' HFTOKEN='${HF_TOKEN:-}' NGPU='${NGPU:-2}'"

cat <<EOF

Submitted from $REMOTE_DIR. Useful follow-ups:
  ssh $CHTC_USER@$CHTC_SERVER 'condor_watch_q'
  ssh $CHTC_USER@$CHTC_SERVER 'condor_q -better-analyze <jobid>'   # if it sits idle
  rsync -e "ssh -o ControlPath=$CONTROL_PATH" -av \\
      $CHTC_USER@$CHTC_SERVER:$REMOTE_DIR/ ./chtc-runs/$RUN_ID/     # pull results
EOF
