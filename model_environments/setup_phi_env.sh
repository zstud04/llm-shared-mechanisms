#!/bin/bash
# Build the Phi environment from phi_env.yml, then install a prebuilt flash-attn
# wheel matching its torch (2.5.1) / CUDA (cu121) / python (cp311) build.
#
# Mirrors setup_gemma4_env.sh: the env is created by --prefix under $CACHE_DIR
# so it lands on project storage, and credentials come from the repo-root .env.

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_YML="${REPO_ROOT}/model_environments/phi_env.yml"

if [ -f "${REPO_ROOT}/.env" ]; then
    echo "Loading credentials from .env..."
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.env"
fi

if [ -z "${CACHE_DIR:-}" ]; then
    echo "CACHE_DIR is unset. Set it (or add it to .env) so the env is not"
    echo "created in \$HOME; it points at project storage."
    exit 1
fi

command -v conda >/dev/null || { echo "Conda not found."; exit 1; }

ENV_NAME=$(grep "^name:" "$ENV_YML" | head -n1 | cut -d " " -f 2)
CONDA_BASE="$(conda info --base)"
case "$CONDA_BASE" in
    "$CACHE_DIR"*) ENVS_DIR="${CONDA_BASE}/envs" ;;
    *)             ENVS_DIR="${CACHE_DIR}/conda_envs" ;;
esac
ENV_PREFIX="${ENVS_DIR}/${ENV_NAME}"

if [ -d "$ENV_PREFIX" ]; then
    echo "Environment already exists at $ENV_PREFIX"
else
    echo "Creating '$ENV_NAME' at $ENV_PREFIX ..."
    mkdir -p "$ENVS_DIR"
    conda env create -f "$ENV_YML" --prefix "$ENV_PREFIX"
fi

# flash-attn: install the prebuilt wheel for torch2.5 / cu12 / cp311. Building
# from source is avoided (no nvcc); --no-build-isolation keeps pip from pulling
# a fresh torch. If the pinned wheel 404s, bump the version here.
FA_WHL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
echo "Installing flash-attn from $FA_WHL"
"${ENV_PREFIX}/bin/pip" install --no-build-isolation "$FA_WHL"

if [ -n "${HF_TOKEN:-}" ]; then
    "${ENV_PREFIX}/bin/hf" auth login --token "$HF_TOKEN" 2>/dev/null \
        || "${ENV_PREFIX}/bin/huggingface-cli" login --token "$HF_TOKEN" || true
fi

echo
echo "Done. Activate with:  conda activate ${ENV_PREFIX}"
"${ENV_PREFIX}/bin/python" -c "import torch, flash_attn; print('torch', torch.__version__, 'flash_attn', flash_attn.__version__)"
