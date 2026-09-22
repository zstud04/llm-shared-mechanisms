# Running experiments on CHTC

Everything here submits `script/exp_battery.py` to UW-Madison CHTC via
HTCondor. Files:

| file | purpose |
| --- | --- |
| `deploy.sh` | bundle the repo, push to an Access Point, submit |
| `experiment.sh` | in-container job wrapper (caches, GPU monitor, run, collect) |
| `gpu_monitor.sh` | samples `nvidia-smi` into `gpu_metrics.csv` |
| `exp_run.sub` | one GPU — models that fit a single card |
| `exp_run_multigpu.sub` | sharded across GPUs — models >30B |
| `exp_run_sweep.sub` | many one-GPU jobs from `params.txt` |

## Quick start

```bash
export CHTC_USER=<your netid>
chtc/deploy.sh -- general_eval gemma-2-9b-it \
    stimuli/general_eval.csv general_instruct data/model/behavioral
```

Submit from `/home` on the Access Point (this is what `deploy.sh` does). Keep
submit files and code in `/home`; use `$STAGING` only for large containers and
datasets.

## Choosing a submit file

Start with the least restrictive shape that can do the work, then tighten it
only when a pilot job proves you need to.

**One GPU (`exp_run.sub`)** — any model at or below ~30B in bf16. This matches
the most slots and starts fastest.

**Multiple GPUs (`exp_run_multigpu.sub`)** — models above ~30B. Sharding is
automatic: `utils/model_utils.load_model` reads `n_params_b` from
`config/models.json`, and past `SHARD_THRESHOLD_B` (30) it turns on
`device_map="auto"` for HF, `n_devices` for TransformerLens, and
`tensor_parallel_size` for vLLM, each sized to the GPUs the job was allocated.
Nothing model-specific needs editing — just `request_gpus`.

**Many one-GPU jobs (`exp_run_sweep.sub`)** — the preferred way to run the
ablation sweep, which is the slowest experiment (one forward pass per
component per prompt). Split it by category with `--subset-categories`; each
job writes a separately-suffixed CSV that concatenates afterwards. Several
short jobs match sooner than one long multi-GPU job, and `+is_resumable = true`
opens backfill capacity.

Prefer `--method attribution` before reaching for more GPUs: it replaces
`n_layers x n_heads` forward passes per prompt with a single forward+backward
pass, which is usually a much bigger win than extra hardware.

## Before scaling up

1. Run one pilot job and read the `.log` file. It reports requested vs. used
   CPU, memory, and disk — right-size the request from that, because oversized
   requests reduce matching and lengthen queue time.
2. Check `gpu_metrics.csv`. Low GPU utilization with busy CPUs means the
   bottleneck is data handling, not the card; low utilization with idle CPUs
   means more CPUs will not help.
3. If a job sits idle, run `condor_q -better-analyze <jobid>` before changing
   anything. Loosen `gpus_minimum_memory`, drop `request_memory`, or shorten
   `+GPUJobLength` before requesting different hardware.

## Runtime classes

- `+GPUJobLength = "short"` for anything under 12h — the fastest-starting class.
- `"medium"` for 12–24h.
- `"long"` (up to 7 days) only when uninterrupted runtime genuinely matters;
  its per-user concurrency limit is much lower.

Do not set `ConcurrencyLimits` by hand; CHTC's submit transforms assign them.

## Retrieving results

`experiment.sh` copies everything under `data/model/` into `result/`, which
comes back with `transfer_output_files`. Pull a finished run with:

```bash
rsync -av <netid>@ap2002.chtc.wisc.edu:chtc-runs/<RUN_ID>/ ./chtc-runs/<RUN_ID>/
```

Then move the CSVs into `data/model/<experiment>/` in the repo.

## Container

The submit files use `docker://studdiford/causal:v8`. When dependencies change
(this branch adds vLLM and accelerate), build a **new tag** and update the
submit files — never overwrite an existing tag or OSDF path, since cached
layers may be reused. Verify a new image with a smoke test that prints
`torch.__version__`, `torch.cuda.is_available()`, and `torch.cuda.device_count()`
before launching real work.
