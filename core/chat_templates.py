"""Instruct-model prompt wrappers.

Which template a model gets is read from the `chat_template` field in
`config/models.json` (null for base models). The old code sniffed the model
path for "it"/"chat" substrings, which mislabelled paths like
`meta-llama/Llama-2-7b-hf`.
"""

GEMMA_TEMPLATE = (
    "<bos><start_of_turn>user\n"
    "{instruction}"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)

# Gemma 4 replaced the <start_of_turn>/<end_of_turn> markers with
# <|turn>...<turn|>, so GEMMA_TEMPLATE is wrong for it. Both variants below are
# what google/gemma-4-*-it tokenizers render for a single user message; they are
# identical across the 12B, 31B and 26B-A4B checkpoints.
#
# The default closes an empty "thought" channel, which suppresses the CoT and
# makes the next token the answer -- what the logit-based measures need.
GEMMA4_TEMPLATE = (
    "<bos><|turn>user\n"
    "{instruction}"
    "<turn|>\n"
    "<|turn>model\n"
    "<|channel>thought\n<channel|>"
)

# Thinking enabled (tokenizer's enable_thinking=True): use for CoT runs, where
# the trace is extracted from the generation rather than skipped.
GEMMA4_THINK_TEMPLATE = (
    "<bos><|turn>system\n"
    "<|think|>\n"
    "<turn|>\n"
    "<|turn>user\n"
    "{instruction}"
    "<turn|>\n"
    "<|turn>model\n"
)

LLAMA_TEMPLATE = "<s>[INST] {instruction} [/INST]"

# Llama-3 / 3.1 header format. The existing llama-3.1 entries use the llama-2
# "llama" key for backward compatibility; new Llama-3 instruct models use this.
LLAMA3_TEMPLATE = (
    "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)

# Llama-4 renamed the header markers to <|header_start|>/<|header_end|> and the
# turn terminator to <|eot|>.
LLAMA4_TEMPLATE = (
    "<|begin_of_text|><|header_start|>user<|header_end|>\n\n"
    "{instruction}<|eot|>"
    "<|header_start|>assistant<|header_end|>\n\n"
)

QWEN_TEMPLATE = (
    "<|im_start|>user\n"
    "{instruction}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Phi-3 / Phi-3.5 chat markers.
PHI3_TEMPLATE = (
    "<|user|>\n"
    "{instruction}<|end|>\n"
    "<|assistant|>\n"
)

# OLMo-2 / OLMo-3 chat markers.
OLMO_TEMPLATE = (
    "<|endoftext|><|user|>\n"
    "{instruction}\n"
    "<|assistant|>\n"
)

# OpenAI gpt-oss "harmony" response format. The assistant turn opens the final
# channel so the next token is the answer rather than an analysis trace.
HARMONY_TEMPLATE = (
    "<|start|>user<|message|>{instruction}<|end|>"
    "<|start|>assistant<|channel|>final<|message|>"
)

BASE_TEMPLATE = (
    "You are a helpful assistant who answers questions."
    "{instruction} "
    "The answer is "
)

TEMPLATES = {
    "gemma": GEMMA_TEMPLATE,
    "gemma4": GEMMA4_TEMPLATE,
    "gemma4_think": GEMMA4_THINK_TEMPLATE,
    "llama": LLAMA_TEMPLATE,
    "llama3": LLAMA3_TEMPLATE,
    "llama4": LLAMA4_TEMPLATE,
    "qwen": QWEN_TEMPLATE,
    "phi3": PHI3_TEMPLATE,
    "olmo": OLMO_TEMPLATE,
    "harmony": HARMONY_TEMPLATE,
    "base": BASE_TEMPLATE,
}


def template_for(model_cfg: dict):
    """Return the template string for a model config, or None for base models."""
    key = model_cfg.get("chat_template")
    if not key:
        return None
    if key not in TEMPLATES:
        raise KeyError(f"Unknown chat_template {key!r}; known: {sorted(TEMPLATES)}")
    return TEMPLATES[key]
