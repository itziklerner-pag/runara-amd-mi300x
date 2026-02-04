# Runara: AMD MI300X Llama-3.3-70B FP8 Benchmark Project

## Executive Summary

This document outlines a plan to benchmark Llama-3.3-70B FP8 on AMD MI300X GPU via AMD Developer Cloud (powered by DigitalOcean), with automated instance management to prevent cost overruns.

---

## 1. Feasibility Analysis

### 1.1 Memory Requirements: Can 70B FP8 Run on 1x MI300X?

**YES - Llama-3.3-70B FP8 fits comfortably on a single MI300X.**

| Model | Precision | VRAM Required | MI300X VRAM | Headroom |
|-------|-----------|---------------|-------------|----------|
| Llama-3.3-70B | BF16 | ~140 GB | 192 GB HBM3 | 52 GB |
| Llama-3.3-70B | FP8 | ~70-80 GB | 192 GB HBM3 | **112+ GB** |

**Calculation:**
- 70B parameters × 1 byte (FP8) = 70 GB base model
- KV cache overhead for 128k context ≈ 20-40 GB (variable)
- Total: ~90-110 GB maximum during inference
- **MI300X 192 GB provides ample headroom**

### 1.2 Why MI300X for This Benchmark?

| Feature | MI300X | H100 (80GB) | Notes |
|---------|--------|-------------|-------|
| VRAM | 192 GB HBM3 | 80 GB HBM3 | MI300X: 2.4x more |
| Memory BW | 5.3 TB/s | 3.35 TB/s | MI300X: 60% faster |
| Can run 70B FP8 on 1 GPU? | ✅ Yes | ⚠️ Tight | Major advantage |
| Cost (DO Gradient) | ~$2/hr | ~$2.50/hr | MI300X better value |

---

## 2. AMD Developer Cloud / DigitalOcean Setup

### 2.1 Understanding the Platform

AMD Developer Cloud is **hosted on DigitalOcean** (branded as "DigitalOcean Gradient AI"). You interact with it via:
- **DigitalOcean Control Panel**: https://cloud.digitalocean.com
- **doctl CLI**: DigitalOcean's official CLI
- **DigitalOcean API**: RESTful API at https://api.digitalocean.com/v2/

### 2.2 Available AMD GPU Droplet Plans

| GPU | Slug | GPU VRAM | RAM | vCPUs | Boot Disk | Scratch Disk |
|-----|------|----------|-----|-------|-----------|--------------|
| MI300X (1x) | `gpu-mi300x1-192gb` | 192 GB | 240 GiB | 20 | 720 GiB | 5 TiB |
| MI300X (8x) | `gpu-mi300x8-1536gb` | 1,536 GB | 1,920 GiB | 160 | 2,046 GiB | 40 TiB |

**For our benchmark: Use `gpu-mi300x1-192gb`** (single GPU is sufficient for 70B FP8)

### 2.3 Available Quickstart Images

Based on the AMD Developer Cloud listing, these quickstart images are available:

| Image | Use Case | Recommended? |
|-------|----------|--------------|
| **vLLM 0.9.2** | LLM inference serving | ⭐ **YES - Best for benchmark** |
| SGLang 0.4.9 | LLM inference (alternative) | ✅ Good alternative |
| Megatron 0.10.0 | Training | ❌ Not for inference |
| JAX 0.4.35 | General ML | ❌ Not optimized for LLM |
| ROCm GPT-OSS | Development | ❌ Requires setup |
| Base ROCm 7.1 | Raw environment | ❌ Requires manual install |

**Recommendation: Use vLLM 0.9.2 quickstart image**

Alternatively, DigitalOcean provides an AI/ML-ready base image:
- **Slug**: `gpu-amd-base`
- **Includes**: Ubuntu 24.04, ROCm 6.4.0, amdgpu-dkms 6.12.12

---

## 3. Programmatic Instance Management

### 3.1 Setup: Install and Configure doctl

```bash
# Install doctl
sudo snap install doctl  # or brew install doctl

# Authenticate
doctl auth init
# Enter your DigitalOcean API token

# Verify
doctl account get
```

### 3.2 Create MI300X GPU Droplet

```bash
# Using vLLM quickstart image (if available as slug)
doctl compute droplet create runara-benchmark \
  --size gpu-mi300x1-192gb \
  --image gpu-amd-base \
  --region tor1 \
  --ssh-keys <your-ssh-key-id> \
  --tag-name runara \
  --tag-name auto-destroy \
  --wait

# Get droplet ID and IP
doctl compute droplet list --tag-name runara --format ID,Name,PublicIPv4,Status
```

### 3.3 API Method (Python)

```python
import os
from pydo import Client

client = Client(token=os.environ.get("DIGITALOCEAN_TOKEN"))

# Create GPU Droplet
req = {
    "name": "runara-benchmark",
    "region": "tor1",  # Toronto has AMD GPUs
    "size": "gpu-mi300x1-192gb",
    "image": "gpu-amd-base",
    "ssh_keys": [os.environ.get("DO_SSH_KEY_ID")],
    "tags": ["runara", "auto-destroy"],
    "monitoring": True,
}

resp = client.droplets.create(body=req)
droplet_id = resp["droplet"]["id"]
print(f"Created droplet: {droplet_id}")
```

### 3.4 Destroy Instance

```bash
# By ID
doctl compute droplet delete 123456789 --force

# By tag (safer - destroy all tagged instances)
doctl compute droplet delete --tag-name runara --force
```

---

## 4. Benchmark Script Design

### 4.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     runara-benchmark.sh                          │
├─────────────────────────────────────────────────────────────────┤
│  1. Create Droplet (doctl)                                      │
│  2. Wait for SSH access                                         │
│  3. SSH: Setup vLLM + Download model                            │
│  4. SSH: Run benchmark suite                                    │
│  5. SSH: Collect results                                        │
│  6. SCP: Download results                                       │
│  7. Destroy Droplet (doctl)                                     │
│  8. Always destroy on exit (trap)                               │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 Main Orchestration Script

```bash
#!/bin/bash
# runara-benchmark.sh - Main orchestration script

set -euo pipefail

# Configuration
DROPLET_NAME="runara-$(date +%s)"
SIZE="gpu-mi300x1-192gb"
REGION="tor1"
IMAGE="gpu-amd-base"
MODEL="meta-llama/Llama-3.3-70B-Instruct"
QUANTIZATION="fp8"
MAX_RUNTIME_SECONDS=7200  # 2 hours max
SSH_KEY_ID="${DO_SSH_KEY_ID}"
RESULTS_DIR="./results/$(date +%Y%m%d-%H%M%S)"

# Cleanup function
cleanup() {
    echo "🧹 Cleaning up..."
    if [[ -n "${DROPLET_ID:-}" ]]; then
        echo "Destroying droplet $DROPLET_ID..."
        doctl compute droplet delete "$DROPLET_ID" --force || true
    fi
    # Kill timeout watchdog if running
    [[ -n "${WATCHDOG_PID:-}" ]] && kill "$WATCHDOG_PID" 2>/dev/null || true
}

# Always cleanup on exit
trap cleanup EXIT INT TERM

# Start timeout watchdog
start_watchdog() {
    (
        sleep $MAX_RUNTIME_SECONDS
        echo "⚠️ MAX RUNTIME EXCEEDED - Force destroying instance"
        doctl compute droplet delete --tag-name "$DROPLET_NAME" --force 2>/dev/null || true
        exit 1
    ) &
    WATCHDOG_PID=$!
}

echo "🚀 Starting Runara AMD GPU Benchmark"
echo "Model: $MODEL ($QUANTIZATION)"
mkdir -p "$RESULTS_DIR"

# Start watchdog
start_watchdog

# Create droplet
echo "📦 Creating GPU droplet..."
doctl compute droplet create "$DROPLET_NAME" \
    --size "$SIZE" \
    --image "$IMAGE" \
    --region "$REGION" \
    --ssh-keys "$SSH_KEY_ID" \
    --tag-name "$DROPLET_NAME" \
    --wait \
    --format ID,PublicIPv4 \
    --no-header > /tmp/droplet_info.txt

DROPLET_ID=$(cut -f1 -d' ' /tmp/droplet_info.txt)
DROPLET_IP=$(cut -f2 -d' ' /tmp/droplet_info.txt)
echo "✅ Created droplet $DROPLET_ID at $DROPLET_IP"

# Wait for SSH
echo "⏳ Waiting for SSH access..."
for i in {1..60}; do
    if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@"$DROPLET_IP" "echo connected" 2>/dev/null; then
        echo "✅ SSH connected"
        break
    fi
    echo "  Attempt $i/60..."
    sleep 10
done

# Run remote benchmark
echo "🏃 Running benchmark on remote GPU..."
ssh root@"$DROPLET_IP" 'bash -s' < ./scripts/remote-benchmark.sh 2>&1 | tee "$RESULTS_DIR/benchmark.log"

# Download results
echo "📥 Downloading results..."
scp -r root@"$DROPLET_IP":/root/benchmark-results/* "$RESULTS_DIR/"

echo "✅ Benchmark complete. Results in: $RESULTS_DIR"
echo "💰 Droplet will be destroyed now to prevent charges"
# cleanup() will be called by trap
```

### 4.3 Remote Benchmark Script (runs on GPU instance)

```bash
#!/bin/bash
# scripts/remote-benchmark.sh - Runs on the MI300X instance

set -euo pipefail

MODEL="meta-llama/Llama-3.3-70B-Instruct"
QUANTIZATION="fp8"
RESULTS_DIR="/root/benchmark-results"
mkdir -p "$RESULTS_DIR"

echo "=== GPU Info ==="
rocm-smi
rocminfo | head -50

echo "=== Installing vLLM for AMD ==="
pip install vllm  # Assumes ROCm-compatible vLLM is available

# Or use Docker (recommended for reproducibility)
# docker pull rocm/vllm:latest
# docker run --device=/dev/kfd --device=/dev/dri \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     rocm/vllm:latest ...

echo "=== Downloading Model ==="
# Authenticate with HuggingFace
export HF_TOKEN="${HF_TOKEN:-}"
huggingface-cli login --token "$HF_TOKEN"

echo "=== Starting vLLM Server ==="
# Start server in background for benchmark
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --quantization "$QUANTIZATION" \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port 8000 &

VLLM_PID=$!
sleep 60  # Wait for model to load

echo "=== Running Benchmark Suite ==="

# Benchmark 1: Throughput test (long prompt, many tokens)
echo "--- Throughput Test ---"
python << 'EOF'
import requests
import time
import json

results = {"tests": []}

# Throughput test: Generate many tokens
prompt = "Write a comprehensive essay about the history of artificial intelligence, covering its origins, major milestones, current state, and future prospects. Include discussion of neural networks, machine learning, deep learning, and large language models."

start = time.time()
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.7,
    }
)
elapsed = time.time() - start
data = response.json()
tokens = data["usage"]["completion_tokens"]

results["tests"].append({
    "name": "throughput_2048_tokens",
    "tokens_generated": tokens,
    "time_seconds": elapsed,
    "tokens_per_second": tokens / elapsed,
})
print(f"Throughput: {tokens / elapsed:.2f} tokens/sec")

# Latency test: Time to first token (short response)
prompt = "What is 2+2?"
start = time.time()
response = requests.post(
    "http://localhost:8000/v1/completions",
    json={
        "model": "meta-llama/Llama-3.3-70B-Instruct",
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0,
    },
    stream=True
)
# For streaming TTFT measurement
for chunk in response.iter_content(chunk_size=1):
    first_token_time = time.time() - start
    break
    
results["tests"].append({
    "name": "ttft_short_prompt",
    "time_to_first_token_ms": first_token_time * 1000,
})
print(f"TTFT: {first_token_time * 1000:.2f} ms")

# Concurrent request test
import concurrent.futures

def make_request(i):
    start = time.time()
    response = requests.post(
        "http://localhost:8000/v1/completions",
        json={
            "model": "meta-llama/Llama-3.3-70B-Instruct",
            "prompt": f"Write a short poem about number {i}.",
            "max_tokens": 100,
            "temperature": 0.7,
        }
    )
    return time.time() - start, response.json()["usage"]["completion_tokens"]

concurrent_results = []
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    start = time.time()
    futures = [executor.submit(make_request, i) for i in range(10)]
    for future in concurrent.futures.as_completed(futures):
        concurrent_results.append(future.result())
    total_time = time.time() - start

total_tokens = sum(r[1] for r in concurrent_results)
results["tests"].append({
    "name": "concurrent_10_requests",
    "total_tokens": total_tokens,
    "total_time_seconds": total_time,
    "effective_throughput": total_tokens / total_time,
})
print(f"Concurrent throughput: {total_tokens / total_time:.2f} tokens/sec")

# Save results
with open("/root/benchmark-results/benchmark_results.json", "w") as f:
    json.dump(results, f, indent=2)
print("Results saved!")
EOF

# Collect GPU metrics during benchmark
echo "=== GPU Utilization During Benchmark ==="
rocm-smi --showmeminfo vram --showuse --showpower | tee "$RESULTS_DIR/gpu_metrics.txt"

# Kill vLLM server
kill $VLLM_PID 2>/dev/null || true

echo "=== Benchmark Complete ==="
ls -la "$RESULTS_DIR"
```

---

## 5. NVIDIA-Comparable Result Format

### 5.1 Standard Benchmark Metrics

```json
{
  "metadata": {
    "timestamp": "2025-01-31T04:30:00Z",
    "gpu": "AMD Instinct MI300X",
    "gpu_vram_gb": 192,
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "quantization": "FP8",
    "framework": "vLLM 0.9.2",
    "rocm_version": "6.4.0"
  },
  "benchmarks": {
    "throughput": {
      "tokens_per_second": 45.2,
      "test_config": {
        "max_tokens": 2048,
        "batch_size": 1,
        "prompt_length": 256
      }
    },
    "latency": {
      "time_to_first_token_ms": 125.4,
      "inter_token_latency_ms": 22.1
    },
    "concurrent": {
      "10_requests_tokens_per_second": 312.5,
      "32_requests_tokens_per_second": 445.8
    },
    "memory": {
      "vram_used_gb": 85.2,
      "vram_peak_gb": 92.1,
      "vram_utilization_pct": 48.0
    }
  },
  "comparable_nvidia": {
    "h100_equivalent_throughput_ratio": 0.95,
    "notes": "MI300X achieves ~95% of H100 throughput for 70B models while offering 2.4x VRAM"
  }
}
```

### 5.2 Comparison Table Template

```markdown
| Metric | MI300X (FP8) | H100 (FP8) | Notes |
|--------|--------------|------------|-------|
| Throughput (tok/s) | TBD | ~47 | Single user |
| TTFT (ms) | TBD | ~120 | First token |
| Batch Throughput | TBD | ~480 | 32 concurrent |
| VRAM Used | TBD | ~75 GB | |
| Power Draw | TBD | ~700W | |
```

---

## 6. Cost Estimation & Safeguards

### 6.1 Pricing (DigitalOcean Gradient AI)

| Resource | Price | Notes |
|----------|-------|-------|
| MI300X 1x GPU | ~$2.00/hr | Estimated, verify current pricing |
| MI300X 8x GPU | ~$16.00/hr | For scale-out tests |
| H100 1x GPU | ~$2.50/hr | For comparison |

### 6.2 Estimated Benchmark Costs

| Scenario | Time | Cost |
|----------|------|------|
| Quick benchmark | 30 min | ~$1.00 |
| Full benchmark suite | 2 hours | ~$4.00 |
| Extended testing | 4 hours | ~$8.00 |

### 6.3 Cost Safeguards

#### A. Hard Timeout (Critical)
```bash
# In orchestration script
MAX_RUNTIME_SECONDS=7200  # 2 hours absolute max
(sleep $MAX_RUNTIME_SECONDS && doctl compute droplet delete --tag-name runara --force) &
```

#### B. Trap-Based Cleanup
```bash
trap 'doctl compute droplet delete --tag-name runara --force' EXIT INT TERM ERR
```

#### C. Tag-Based Cleanup Script
```bash
#!/bin/bash
# cleanup-runara.sh - Emergency cleanup
echo "Destroying all Runara instances..."
doctl compute droplet list --tag-name runara --format ID --no-header | \
    xargs -I {} doctl compute droplet delete {} --force
echo "Done."
```

#### D. DigitalOcean Billing Alerts
```bash
# Set up via API or control panel
# Alert at $10, $25, $50 thresholds
```

#### E. Instance Monitoring Cron
```bash
# Add to crontab on a control machine
*/30 * * * * /path/to/check-runara-instances.sh

# check-runara-instances.sh
#!/bin/bash
COUNT=$(doctl compute droplet list --tag-name runara --format ID --no-header | wc -l)
if [ "$COUNT" -gt 0 ]; then
    echo "WARNING: $COUNT Runara instances still running!" | mail -s "Runara Alert" you@email.com
fi
```

---

## 7. Complete Implementation Checklist

### Phase 1: Setup
- [ ] Create DigitalOcean account
- [ ] Generate API token with write access
- [ ] Install and authenticate doctl
- [ ] Set up SSH key in DigitalOcean
- [ ] Get Hugging Face token (for Llama access)
- [ ] Accept Llama 3.3 license on Hugging Face

### Phase 2: Scripts
- [ ] Create `runara-benchmark.sh` (main orchestrator)
- [ ] Create `scripts/remote-benchmark.sh` (GPU runner)
- [ ] Create `scripts/cleanup-runara.sh` (emergency stop)
- [ ] Test with a quick dry-run (short benchmark)

### Phase 3: Execution
- [ ] Run full benchmark
- [ ] Collect and analyze results
- [ ] Compare with published H100 numbers
- [ ] Document findings

### Phase 4: Cleanup
- [ ] Verify all instances destroyed
- [ ] Check DigitalOcean billing
- [ ] Archive results

---

## 8. Quick Start Commands

```bash
# 1. Set environment variables
export DIGITALOCEAN_TOKEN="your-do-token"
export DO_SSH_KEY_ID="your-ssh-key-id"
export HF_TOKEN="your-huggingface-token"

# 2. Test connection
doctl account get

# 3. List available GPU sizes
doctl compute size list | grep gpu

# 4. Run benchmark
./runara-benchmark.sh

# 5. Emergency cleanup
./scripts/cleanup-runara.sh
```

---

## 9. Notes on Quickstart Images

The AMD Developer Cloud mentions these quickstart options. Their exact image slugs need to be discovered at runtime:

```bash
# List available images
doctl compute image list --public | grep -i amd
doctl compute image list --public | grep -i vllm
doctl compute image list --public | grep -i rocm
```

If vLLM quickstart image is available, use it directly. Otherwise, use `gpu-amd-base` and install vLLM manually:

```bash
# On the MI300X instance
pip install vllm  # ROCm-compatible build
# or
docker pull rocm/vllm:latest
```

---

## 10. Appendix: ROCm vs CUDA Notes for Benchmark Comparability

| NVIDIA | AMD | Notes |
|--------|-----|-------|
| CUDA | ROCm | Runtime environment |
| cuDNN | MIOpen | Neural network primitives |
| TensorRT | Composable Kernel | Optimization library |
| nvcc | hipcc | Compiler |
| nvidia-smi | rocm-smi | Monitoring |

vLLM abstracts most differences, making benchmarks directly comparable at the application level.

---

*Document created: 2025-01-31*
*Project: Runara AMD GPU Benchmarking*
