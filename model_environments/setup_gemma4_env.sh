#!/bin/bash
# Build the Gemma-4 environment from gemma4_env.yml.
#
# The env is created by *prefix* under $CACHE_DIR so it lands on project
# storage rather than in $HOME (which is small and shared). Credentials are
# read from the repo-root .env, same as setup_env.sh.

set -eu

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_YML="${REPO_ROOT}/model_environments/gemma4_env.yml"

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

if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install Conda and retry."
    exit 1
fi

ENV_NAME=$(grep "^name:" "$ENV_YML" | head -n1 | cut -d " " -f 2)
# Reuse the conda installation's existing envs/ dir when it is already under
# CACHE_DIR; otherwise fall back to $CACHE_DIR/conda_envs.
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

if [ -n "${HF_TOKEN:-}" ]; then
    echo "Logging in to Hugging Face (Gemma is gated)..."
    "${ENV_PREFIX}/bin/hf" auth login --token "$HF_TOKEN" 2>/dev/null \
        || "${ENV_PREFIX}/bin/huggingface-cli" login --token "$HF_TOKEN"
fi

echo
echo "Done. Activate with:"
echo "    conda activate ${ENV_PREFIX}"
