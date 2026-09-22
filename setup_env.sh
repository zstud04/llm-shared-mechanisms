#!/bin/bash
# Create the conda environment and collect credentials.
#
# Variable names must match what utils/model_utils.py reads:
#   HF_TOKEN      Hugging Face (gated models: Gemma, Llama)
#   OPENAI_KEY    OpenAI
#   GEMINI_KEY    Google Gemini
#   DEEPSEEK_KEY  DeepSeek
#   CACHE_DIR     where model weights are cached

set -u

if [ -f .env ]; then
    echo "Loading credentials from .env..."
    # shellcheck disable=SC1091
    source .env
else
    echo "No .env found. Enter your credentials (blank to skip any of them)."
    read -rp "HF_TOKEN: "     HF_TOKEN
    read -rp "OPENAI_KEY: "   OPENAI_KEY
    read -rp "GEMINI_KEY: "   GEMINI_KEY
    read -rp "DEEPSEEK_KEY: " DEEPSEEK_KEY
    read -rp "CACHE_DIR (model cache path): " CACHE_DIR

    cat <<EOF > .env
export HF_TOKEN="${HF_TOKEN}"
export OPENAI_KEY="${OPENAI_KEY}"
export GEMINI_KEY="${GEMINI_KEY}"
export DEEPSEEK_KEY="${DEEPSEEK_KEY}"
export CACHE_DIR="${CACHE_DIR}"
EOF
    chmod 600 .env
    echo ".env created (gitignored); it will be loaded automatically next time."
fi

ENV_YML='./environment.yml'

if ! command -v conda &> /dev/null; then
    echo "Conda could not be found. Please install Conda and retry."
    exit 1
fi

if [ ! -f "$ENV_YML" ]; then
    echo "No environment.yml file found at $ENV_YML"
    exit 1
fi

ENV_NAME=$(grep "^name:" "$ENV_YML" | head -n1 | cut -d " " -f 2)
echo "Environment name from $ENV_YML: $ENV_NAME"

if [ "${CONDA_DEFAULT_ENV:-}" = "$ENV_NAME" ]; then
    echo "Environment '$ENV_NAME' is already activated."
elif conda env list | grep -qE "^[^#]*$ENV_NAME(\s|$)"; then
    echo "Environment '$ENV_NAME' already exists. Activating it..."
    conda activate "$ENV_NAME"
else
    echo "Creating '$ENV_NAME' from $ENV_YML..."
    conda env create -f "$ENV_YML"
    conda activate "$ENV_NAME"
fi

echo "Registering the R kernel with Jupyter..."
R -q -e "IRkernel::installspec(name = 'r-${ENV_NAME}', displayname = 'R (${ENV_NAME})')"

if [ -n "${HF_TOKEN:-}" ]; then
    echo "Logging in to Hugging Face..."
    hf auth login --token "$HF_TOKEN" 2>/dev/null || huggingface-cli login --token "$HF_TOKEN"
fi

echo "Setup complete. Environment '$ENV_NAME' is ready."
