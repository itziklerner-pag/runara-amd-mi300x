# Runara AMD MI300X Benchmark

Benchmark Llama-3.3-70B FP8 on AMD MI300X, with results in NVIDIA-compatible format.

## Features

- **Persistent Volume** — Model weights cached, survives droplet destruction
- **Snapshot** — vLLM pre-installed image, faster startup
- **Auto-destroy** — Droplet killed after benchmark to prevent cost leaks
- **NVIDIA-compatible output** — Same format for easy comparison

## First Run (Setup)

First run creates snapshot + downloads model (~30-45 min, ~$1-2):

```bash
# 1. Set credentials
export DIGITALOCEAN_TOKEN="your-token"
doctl auth init  # paste token when prompted

export HF_TOKEN="your-huggingface-token"

# 2. Run setup (creates snapshot + downloads model)
cd ~/clawd/projects/runara
./scripts/runara-benchmark.sh --setup-only
```

This creates:
- **Snapshot**: `runara-mi300x-vllm-ready` (vLLM pre-installed)
- **Volume**: `runara-models` (500GB, stores model weights)

## Subsequent Runs (Fast)

After first setup, benchmarks are much faster (~10-15 min):

```bash
# POC - single benchmark (128→2048)
./scripts/runara-benchmark.sh --poc

# Full benchmark (all 9 scenarios)
./scripts/runara-benchmark.sh
```

## Options

| Flag | Description |
|------|-------------|
| `--poc` | Run only first benchmark (128→2048) |
| `--setup-only` | Setup environment, create snapshot, no benchmark |
| `--no-destroy` | Keep droplet running after benchmark (debugging) |

## Emergency Cleanup

```bash
./scripts/cleanup-runara.sh
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    First Run                             │
├─────────────────────────────────────────────────────────┤
│  1. Create droplet from base image                      │
│  2. Create & attach 500GB volume                        │
│  3. Install vLLM                                        │
│  4. Download model to volume                            │
│  5. Create snapshot                                     │
│  6. Destroy droplet (volume persists)                   │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                  Subsequent Runs                         │
├─────────────────────────────────────────────────────────┤
│  1. Create droplet from SNAPSHOT (vLLM ready)           │
│  2. Attach existing volume (model cached)               │
│  3. Run benchmark immediately                           │
│  4. Destroy droplet (volume persists)                   │
└─────────────────────────────────────────────────────────┘
```

## Cost Breakdown

| Resource | Cost | Notes |
|----------|------|-------|
| MI300X Droplet | ~$2/hr | Destroyed after each run |
| 500GB Volume | ~$50/mo | Persists, stores models |
| Snapshot | ~$0.05/GB/mo | Persists, ~20GB |

**First run:** ~$1-2 (setup + snapshot)
**Subsequent POC:** ~$0.50-1 (10-15 min)
**Full benchmark:** ~$2-4 (30-60 min)

## Output

Results saved to `./results/[timestamp]/`:
- `benchmark_nvidia_format.json` - Raw results
- `benchmark_nvidia_format.csv` - NVIDIA-style CSV
- `benchmark.log` - Full execution log

## Test Matrix (NVIDIA Format)

| Input | Output | H200 Reference |
|-------|--------|----------------|
| 128 | 2048 | 4,336 tok/s |
| 128 | 4096 | 2,872 tok/s |
| 2048 | 128 | 442 tok/s |
| 5000 | 500 | 566 tok/s |
| 500 | 2000 | 3,666 tok/s |
| 1000 | 1000 | 2,909 tok/s |
| 1000 | 2000 | 2,994 tok/s |
| 2048 | 2048 | 2,003 tok/s |
| 20000 | 2000 | 283 tok/s |

## Cleanup

To delete persistent resources (when done with project):

```bash
# Delete snapshot
doctl compute snapshot delete $(doctl compute snapshot list --format ID,Name --no-header | grep runara-mi300x | awk '{print $1}')

# Delete volume (WARNING: deletes cached models)
doctl compute volume delete $(doctl compute volume list --format ID,Name --no-header | grep runara-models | awk '{print $1}')
```
