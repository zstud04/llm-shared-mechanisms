# Per-model-family environments

Most models run in the top-level `world_model_env` (`../environment.yml`).
This folder holds extra environments for model families whose dependencies
conflict with it.

## gemma4_env

Gemma 4 (`model_type: gemma4`, `gemma4_unified`) is only implemented in
**transformers >= 5.5**. `world_model_env` sits on the 4.x line because:

- `transformer_lens` 2.x is built against transformers 4.x, and the 3.x line
  that requires transformers >= 5.4 is an API break against
  `core/backends/tlens.py`;
- `vllm` still pins `transformers < 5`.

Upgrading the shared env in place would therefore break the TransformerLens
and vLLM backends for every other model. Gemma 4 gets its own env instead.

Build it:

```bash
bash model_environments/setup_gemma4_env.sh
conda activate "$CACHE_DIR/tmp/miniconda3/envs/gemma4_env"
```

The env is created by `--prefix` under `$CACHE_DIR` so it lands on project
storage rather than in `$HOME`.

### What works for Gemma 4

| Backend  | Status |
| -------- | ------ |
| `hf`     | Supported — inference and interpretability. The default. |
| `tlens`  | Unavailable. TransformerLens has no Gemma-4 conversion (its newest Gemma support is Gemma-3), so every gemma-4 entry in `config/models.json` sets `tl_support: false`. |
| `vllm`   | Unavailable. Not installable alongside transformers >= 5.5. |

Hook sites resolve through the module-tree search in `core/backends/hf.py`,
which finds the text stack underneath the multimodal wrapper without any
per-architecture entry.

For the MoE model (`gemma-4-26b-a4b*`, 128 experts, top-8) the dense
`mlp_pre` / `mlp_post` sites do not exist; `attn_head`, `attn_out`, `mlp_out`
and `resid_post` behave normally. Those entries carry `"moe": true` in
`config/models.json`.
