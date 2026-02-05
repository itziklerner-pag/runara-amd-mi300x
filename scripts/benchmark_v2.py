#!/usr/bin/env python3
"""
Runara Benchmark V2 - Corrected Methodology
- Uses concurrent requests for realistic throughput
- Tracks throughput degradation over output length
- Includes proper cost calculation
- Monitors GPU memory
"""

import subprocess
import requests
import time
import json
import sys
import os
import asyncio
import aiohttp
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading

# Use ungated model - no HF token needed
MODEL = "amd/Llama-3.3-70B-Instruct-FP8-KV"
RESULTS_DIR = "/root/results_v2"
SERVER_URL = "http://localhost:8000"
HOURLY_COST = 2.00  # MI300X on DigitalOcean

# Benchmark configs
CONCURRENCY_LEVELS = [1, 4, 8, 16, 32, 64]
SCENARIOS = [
    (128, 2048),   # Short input, long output
    (128, 4096),   # Short input, very long output
    (2048, 128),   # Long input, short output (summarization)
    (1000, 1000),  # Balanced
    (2048, 2048),  # Both long
]

def generate_prompt(length: int) -> str:
    """Generate a prompt of approximately the specified token length"""
    base = "Explain in detail: "
    repeat = "Continue the analysis with more examples and implications. "
    target_chars = length * 4  # ~4 chars per token approximation
    return (base + repeat * (target_chars // len(repeat)))[:target_chars]

def get_gpu_memory():
    """Get current GPU memory usage via rocm-smi"""
    try:
        result = subprocess.run(
            ['rocm-smi', '--showmeminfo', 'vram', '--json'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
        return None
    except:
        return None

async def run_concurrent_benchmark(input_len, output_len, concurrency, session):
    """Run benchmark with specified concurrency level"""
    prompt = generate_prompt(input_len)
    results = []
    
    async def single_request():
        start = time.time()
        try:
            async with session.post(
                f"{SERVER_URL}/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "max_tokens": output_len,
                    "temperature": 0.7,
                }
            ) as response:
                data = await response.json()
                elapsed = time.time() - start
                tokens = data.get("usage", {}).get("completion_tokens", output_len)
                return {"tokens": tokens, "time": elapsed, "success": True}
        except Exception as e:
            return {"tokens": 0, "time": time.time() - start, "success": False, "error": str(e)}
    
    # Run concurrent requests
    tasks = [single_request() for _ in range(concurrency)]
    results = await asyncio.gather(*tasks)
    
    successful = [r for r in results if r["success"]]
    if not successful:
        return None
    
    total_tokens = sum(r["tokens"] for r in successful)
    total_time = max(r["time"] for r in successful)  # Wall clock time
    throughput = total_tokens / total_time
    
    return {
        "concurrency": concurrency,
        "input_len": input_len,
        "output_len": output_len,
        "total_tokens": total_tokens,
        "wall_time_sec": round(total_time, 2),
        "throughput_tok_s": round(throughput, 1),
        "successful_requests": len(successful),
        "failed_requests": len(results) - len(successful),
    }

async def run_throughput_degradation_test(session):
    """Measure throughput at intervals during long generation"""
    prompt = generate_prompt(128)
    checkpoints = [500, 1000, 2000, 3000, 4000]
    results = []
    
    print("\n📊 Throughput Degradation Test (single stream)")
    print("   Measuring tok/s at different output lengths...")
    
    for target in checkpoints:
        start = time.time()
        try:
            async with session.post(
                f"{SERVER_URL}/v1/completions",
                json={
                    "model": MODEL,
                    "prompt": prompt,
                    "max_tokens": target,
                    "temperature": 0.7,
                }
            ) as response:
                data = await response.json()
                elapsed = time.time() - start
                tokens = data.get("usage", {}).get("completion_tokens", target)
                throughput = tokens / elapsed
                mem = get_gpu_memory()
                
                results.append({
                    "output_tokens": target,
                    "actual_tokens": tokens,
                    "time_sec": round(elapsed, 2),
                    "throughput_tok_s": round(throughput, 1),
                    "gpu_memory": mem,
                })
                print(f"   {target} tokens: {throughput:.1f} tok/s")
        except Exception as e:
            print(f"   {target} tokens: FAILED - {e}")
    
    return results

def calculate_costs(throughput_tok_s):
    """Calculate cost per 1M tokens"""
    tokens_per_hour = throughput_tok_s * 3600
    cost_per_million = (HOURLY_COST / tokens_per_hour) * 1_000_000
    return {
        "hourly_cost_usd": HOURLY_COST,
        "throughput_tok_s": throughput_tok_s,
        "tokens_per_hour": round(tokens_per_hour),
        "cost_per_1m_tokens_usd": round(cost_per_million, 4),
    }

async def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    print("=" * 60)
    print("🚀 Runara Benchmark V2 - Corrected Methodology")
    print("=" * 60)
    print(f"Model: {MODEL}")
    print(f"Concurrency levels: {CONCURRENCY_LEVELS}")
    print()
    
    # Start vLLM server with optimized settings for MI300X
    print("🔧 Starting vLLM server...")
    server = subprocess.Popen([
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", MODEL,
        "--dtype", "auto",
        "--max-model-len", "32768",
        "--gpu-memory-utilization", "0.95",
        "--enable-chunked-prefill",
        "--max-num-seqs", "256",
        "--host", "0.0.0.0",
        "--port", "8000",
    ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    
    # Wait for server to be ready
    print("⏳ Waiting for server to load model...")
    for i in range(180):  # 30 min max wait
        try:
            resp = requests.get(f"{SERVER_URL}/health", timeout=5)
            if resp.status_code == 200:
                print("✅ Server ready!")
                break
        except:
            pass
        if i % 6 == 0:
            print(f"   Still loading... ({i*10}s)")
        time.sleep(10)
    else:
        print("❌ Server failed to start within 30 minutes")
        server.terminate()
        sys.exit(1)
    
    results = {
        "metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "model": MODEL,
            "gpu": "1x AMD MI300X (192GB HBM3)",
            "precision": "FP8-KV (quantized)",
            "framework": "vLLM",
            "benchmark_version": "v2",
            "hourly_cost_usd": HOURLY_COST,
        },
        "concurrency_benchmarks": [],
        "throughput_degradation": [],
        "cost_analysis": {},
    }
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        # Test 1: Concurrency scaling
        print("\n📊 Concurrency Scaling Test")
        print("-" * 40)
        
        for concurrency in CONCURRENCY_LEVELS:
            print(f"\nConcurrency: {concurrency}")
            for input_len, output_len in SCENARIOS:
                result = await run_concurrent_benchmark(input_len, output_len, concurrency, session)
                if result:
                    results["concurrency_benchmarks"].append(result)
                    print(f"  {input_len}→{output_len}: {result['throughput_tok_s']} tok/s")
        
        # Test 2: Throughput degradation
        results["throughput_degradation"] = await run_throughput_degradation_test(session)
        
        # Test 3: Max throughput (for cost calculation) - use highest concurrency
        print("\n📊 Max Throughput Test (concurrency=64)")
        max_result = await run_concurrent_benchmark(128, 2048, 64, session)
        if max_result:
            results["cost_analysis"] = calculate_costs(max_result["throughput_tok_s"])
            print(f"   Max throughput: {max_result['throughput_tok_s']} tok/s")
            print(f"   Cost per 1M tokens: ${results['cost_analysis']['cost_per_1m_tokens_usd']}")
    
    # Save results
    print("\n💾 Saving results...")
    with open(f"{RESULTS_DIR}/benchmark_v2.json", "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 BENCHMARK V2 RESULTS SUMMARY")
    print("=" * 60)
    
    print("\n🔄 Concurrency Scaling (128→2048 scenario):")
    scaling_results = [r for r in results["concurrency_benchmarks"] 
                       if r["input_len"] == 128 and r["output_len"] == 2048]
    for r in scaling_results:
        print(f"   C={r['concurrency']:2d}: {r['throughput_tok_s']:>7.1f} tok/s")
    
    print("\n📉 Throughput Degradation (single stream):")
    for r in results["throughput_degradation"]:
        print(f"   {r['output_tokens']:4d} tokens: {r['throughput_tok_s']:>6.1f} tok/s")
    
    if results["cost_analysis"]:
        print("\n💰 Cost Analysis:")
        ca = results["cost_analysis"]
        print(f"   Max throughput: {ca['throughput_tok_s']} tok/s")
        print(f"   Tokens/hour: {ca['tokens_per_hour']:,}")
        print(f"   Cost per 1M tokens: ${ca['cost_per_1m_tokens_usd']}")
        
        # H200 comparison
        h200_cost_per_1m = 0.50  # Typical H200 cost (higher end)
        savings = ((h200_cost_per_1m - ca['cost_per_1m_tokens_usd']) / h200_cost_per_1m) * 100
        print(f"\n   vs NVIDIA H200 (~$0.50/1M):")
        print(f"      Savings: {savings:.1f}%" if savings > 0 else f"      Premium: {-savings:.1f}%")
    
    server.terminate()
    print("\n✅ Benchmark V2 complete!")
    print(f"📁 Results saved to: {RESULTS_DIR}/benchmark_v2.json")

if __name__ == "__main__":
    asyncio.run(main())
