#!/bin/bash
# =============================================================================
# Remote Benchmark Script - Runs on MI300X Instance
# =============================================================================
# This script runs on the GPU droplet. It:
# 1. Uses cached vLLM (from snapshot) or installs fresh
# 2. Uses cached model (from persistent volume) or downloads
# 3. Runs benchmark suite matching NVIDIA format
# =============================================================================

set -euo pipefail

MODEL="meta-llama/Llama-3.3-70B-Instruct"
RESULTS_DIR="/root/results"
POC_MODE="${POC_MODE:-false}"
export HF_HOME="${HF_HOME:-/mnt/models/huggingface}"

mkdir -p "$RESULTS_DIR"

echo "📁 Model cache: $HF_HOME"
if [[ -d "$HF_HOME/hub" ]]; then
    echo "   Cache size: $(du -sh $HF_HOME/hub 2>/dev/null | cut -f1 || echo 'unknown')"
fi

echo "============================================================"
echo "🔧 MI300X Remote Benchmark Setup"
echo "============================================================"
echo "Model: $MODEL"
echo "POC Mode: $POC_MODE"
echo ""

# -----------------------------------------------------------------------------
# Step 1: GPU Info
# -----------------------------------------------------------------------------
echo "📊 GPU Information:"
echo "-------------------"
rocm-smi || echo "rocm-smi not available yet"
echo ""

# -----------------------------------------------------------------------------
# Step 2: Install vLLM for ROCm
# -----------------------------------------------------------------------------
echo "📦 Installing vLLM for ROCm..."

# Update pip
pip install --upgrade pip

# Install vLLM (ROCm build)
pip install vllm

# Install benchmark dependencies
pip install requests aiohttp

echo "✅ vLLM installed"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Hugging Face Login
# -----------------------------------------------------------------------------
echo "🔑 Authenticating with Hugging Face..."
if [[ -n "${HF_TOKEN:-}" ]]; then
    huggingface-cli login --token "$HF_TOKEN" --add-to-git-credential
    echo "✅ HF authenticated"
else
    echo "⚠️  No HF_TOKEN - model download may fail if gated"
fi
echo ""

# -----------------------------------------------------------------------------
# Step 4: Run Benchmark
# -----------------------------------------------------------------------------
echo "🏃 Starting benchmark..."
echo ""

python3 << 'BENCHMARK_SCRIPT'
import subprocess
import requests
import time
import json
import sys
import os
from datetime import datetime

MODEL = "meta-llama/Llama-3.3-70B-Instruct"
RESULTS_DIR = "/root/results"
POC_MODE = os.environ.get("POC_MODE", "false").lower() == "true"

# NVIDIA-format benchmark scenarios
# Each tuple: (input_length, output_length)
if POC_MODE:
    # POC: Just first scenario
    BENCHMARK_SCENARIOS = [
        (128, 2048),
    ]
    print("🧪 POC MODE: Running single benchmark (128→2048)")
else:
    # Full: All 9 scenarios matching NVIDIA
    BENCHMARK_SCENARIOS = [
        (128, 2048),
        (128, 4096),
        (2048, 128),
        (5000, 500),
        (500, 2000),
        (1000, 1000),
        (1000, 2000),
        (2048, 2048),
        (20000, 2000),
    ]
    print("📊 FULL MODE: Running all 9 benchmark scenarios")

print()

def generate_prompt(length: int) -> str:
    """Generate a prompt of approximately the specified token length."""
    # Approximate: 1 token ≈ 4 characters
    base = "Write a detailed analysis of the following topic. "
    repeat = "Explain the implications and provide examples. "
    target_chars = length * 4
    prompt = base + (repeat * (target_chars // len(repeat)))
    return prompt[:target_chars]

def run_benchmark(input_length: int, output_length: int, server_url: str) -> dict:
    """Run a single benchmark scenario."""
    prompt = generate_prompt(input_length)
    
    print(f"  Running: input={input_length}, output={output_length}...")
    
    # Warmup
    for _ in range(2):
        try:
            requests.post(
                f"{server_url}/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": prompt[:500],
                    "max_tokens": 10,
                    "temperature": 0,
                },
                timeout=120
            )
        except:
            pass
    
    # Actual benchmark (3 runs, take average)
    times = []
    tokens_generated = []
    
    for run in range(3):
        start = time.time()
        try:
            response = requests.post(
                f"{server_url}/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "max_tokens": output_length,
                    "temperature": 0.7,
                },
                timeout=600
            )
            elapsed = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                tokens = data.get("usage", {}).get("completion_tokens", output_length)
                times.append(elapsed)
                tokens_generated.append(tokens)
            else:
                print(f"    Run {run+1}: Error {response.status_code}")
        except Exception as e:
            print(f"    Run {run+1}: Exception {e}")
    
    if times:
        avg_time = sum(times) / len(times)
        avg_tokens = sum(tokens_generated) / len(tokens_generated)
        throughput = avg_tokens / avg_time
    else:
        throughput = 0
        avg_time = 0
        avg_tokens = 0
    
    result = {
        "input_length": input_length,
        "output_length": output_length,
        "throughput_tokens_per_sec": round(throughput, 1),
        "avg_latency_sec": round(avg_time, 2),
        "avg_tokens_generated": round(avg_tokens, 1),
        "runs": len(times),
    }
    
    print(f"    ✅ Throughput: {throughput:.1f} tokens/sec")
    return result

def main():
    print("=" * 60)
    print("🚀 Llama-3.3-70B FP8 Benchmark on AMD MI300X")
    print("=" * 60)
    print()
    
    # Start vLLM server
    print("🔧 Starting vLLM server...")
    print("   Model:", MODEL)
    print("   Quantization: FP8")
    print("   Max model length: 32768")
    print()
    
    server_process = subprocess.Popen(
        [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", MODEL,
            "--quantization", "fp8",
            "--dtype", "float16",
            "--max-model-len", "32768",
            "--gpu-memory-utilization", "0.90",
            "--host", "0.0.0.0",
            "--port", "8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    
    # Wait for server to be ready
    server_url = "http://localhost:8000"
    print("⏳ Waiting for vLLM server to load model...")
    
    for i in range(120):  # 20 minutes max for model loading
        try:
            resp = requests.get(f"{server_url}/health", timeout=5)
            if resp.status_code == 200:
                print("✅ vLLM server ready!")
                break
        except:
            pass
        time.sleep(10)
        if i % 6 == 0:
            print(f"   Still loading... ({i*10}s)")
    else:
        print("❌ vLLM server failed to start")
        server_process.terminate()
        sys.exit(1)
    
    print()
    
    # Run benchmarks
    print("📊 Running benchmarks...")
    print("-" * 40)
    
    results = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": "Llama v3.3 70B",
            "gpu": "1x MI300X",
            "server": "DO Gradient",
            "precision": "FP8",
            "framework": "vLLM",
            "gpu_version": "AMD MI300X",
            "gpu_memory_gb": 192,
        },
        "benchmarks": []
    }
    
    for input_len, output_len in BENCHMARK_SCENARIOS:
        try:
            result = run_benchmark(input_len, output_len, server_url)
            result["pp"] = 1
            result["tp"] = 1
            results["benchmarks"].append(result)
        except Exception as e:
            print(f"  ❌ Failed: {e}")
    
    # Save results
    print()
    print("💾 Saving results...")
    
    # JSON format
    with open(f"{RESULTS_DIR}/benchmark_nvidia_format.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # CSV format (NVIDIA-style)
    with open(f"{RESULTS_DIR}/benchmark_nvidia_format.csv", "w") as f:
        f.write("Model,PP,TP,Input Length,Output Length,Throughput,GPU,Server,Precision,Framework,GPU Version\n")
        for b in results["benchmarks"]:
            f.write(f"Llama v3.3 70B,1,1,{b['input_length']},{b['output_length']},{b['throughput_tokens_per_sec']} output tokens/sec,1x MI300X,DO Gradient,FP8,vLLM,AMD MI300X\n")
    
    # Summary
    print()
    print("=" * 60)
    print("📊 BENCHMARK RESULTS")
    print("=" * 60)
    print()
    print(f"{'Input':<8} {'Output':<8} {'Throughput':<20}")
    print("-" * 40)
    for b in results["benchmarks"]:
        print(f"{b['input_length']:<8} {b['output_length']:<8} {b['throughput_tokens_per_sec']:.1f} tokens/sec")
    print()
    
    # Cleanup
    print("🛑 Stopping vLLM server...")
    server_process.terminate()
    server_process.wait(timeout=30)
    
    print("✅ Benchmark complete!")

if __name__ == "__main__":
    main()
BENCHMARK_SCRIPT

echo ""
echo "✅ Remote benchmark script complete"
