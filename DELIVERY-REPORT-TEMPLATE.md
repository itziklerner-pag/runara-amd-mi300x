# Runara AMD GPU Benchmark Report
## Llama-3.3-70B FP8 Inference Performance: AMD MI300X vs NVIDIA H200

**Report Date:** [DATE]  
**Prepared by:** [YOUR COMPANY]  
**Client:** Runara  
**Classification:** Proprietary - Not to be shared

---

## Executive Summary

This report benchmarks **Llama-3.3-70B FP8** on AMD Instinct MI300X using the identical test methodology and format as [NVIDIA's published benchmarks](https://developer.nvidia.com/deep-learning-performance-training-inference/ai-inference), enabling direct performance comparison.

---

## 1. AMD MI300X Benchmark Results

### Llama v3.3 70B - 1 GPU / FP8 - Max Throughput

| Model | PP | TP | Input Length | Output Length | Throughput | GPU | Server | Precision | Framework | GPU Version |
|-------|----|----|--------------|---------------|------------|-----|--------|-----------|-----------|-------------|
| Llama v3.3 70B | 1 | 1 | 128 | 2048 | **X,XXX output tokens/sec** | 1x MI300X | DO Gradient | FP8 | vLLM 0.9.2 | AMD MI300X |
| Llama v3.3 70B | 1 | 1 | 128 | 4096 | **X,XXX output tokens/sec** | 1x MI300X | DO Gradient | FP8 | vLLM 0.9.2 | AMD MI300X |
| Llama v3.3 70B | 1 | 1 | 2048 | 128 | **XXX output tokens/sec** | 1x MI300X | DO Gradient | FP8 | vLLM 0.9.2 | AMD MI300X |
| Llama v3.3 70B | 1 | 1 | 5000 | 500 | **XXX output tokens/sec** | 1x MI300X | DO Gradient | FP8 | vLLM 0.9.2 | AMD MI300X |
| Llama v3.3 70B | 1 | 1 | 500 | 2000 | **X,XXX output tokens/sec** | 1x MI300X | DO Gradient | FP8 | vLLM 0.9.2 | AMD MI300X |
| Llama v3.3 70B | 1 | 1 | 1000 | 1000 | **X,XXX output tokens/sec** | 1x MI300X | DO Gradient | FP8 | vLLM 0.9.2 | AMD MI300X |
| Llama v3.3 70B | 1 | 1 | 1000 | 2000 | **X,XXX output tokens/sec** | 1x MI300X | DO Gradient | FP8 | vLLM 0.9.2 | AMD MI300X |
| Llama v3.3 70B | 1 | 1 | 2048 | 2048 | **X,XXX output tokens/sec** | 1x MI300X | DO Gradient | FP8 | vLLM 0.9.2 | AMD MI300X |
| Llama v3.3 70B | 1 | 1 | 20000 | 2000 | **XXX output tokens/sec** | 1x MI300X | DO Gradient | FP8 | vLLM 0.9.2 | AMD MI300X |

**Legend:**
- PP: Pipeline Parallelism
- TP: Tensor Parallelism
- Output tokens/second is inclusive of time to generate the first token (tokens/s = total generated tokens / total latency)

---

## 2. NVIDIA H200 Reference Benchmarks

### Llama v3.3 70B - 1 GPU / FP8 - Max Throughput (NVIDIA Published Data)

| Model | PP | TP | Input Length | Output Length | Throughput | GPU | Server | Precision | Framework | GPU Version |
|-------|----|----|--------------|---------------|------------|-----|--------|-----------|-----------|-------------|
| Llama v3.3 70B | 1 | 1 | 128 | 2048 | 4,336 output tokens/sec | 1x H200 | DGX H200 | FP8 | TensorRT-LLM 1.0 | NVIDIA H200 |
| Llama v3.3 70B | 1 | 1 | 128 | 4096 | 2,872 output tokens/sec | 1x H200 | DGX H200 | FP8 | TensorRT-LLM 1.0 | NVIDIA H200 |
| Llama v3.3 70B | 1 | 1 | 2048 | 128 | 442 output tokens/sec | 1x H200 | DGX H200 | FP8 | TensorRT-LLM 1.0 | NVIDIA H200 |
| Llama v3.3 70B | 1 | 1 | 5000 | 500 | 566 output tokens/sec | 1x H200 | DGX H200 | FP8 | TensorRT-LLM 1.0 | NVIDIA H200 |
| Llama v3.3 70B | 1 | 1 | 500 | 2000 | 3,666 output tokens/sec | 1x H200 | DGX H200 | FP8 | TensorRT-LLM 1.0 | NVIDIA H200 |
| Llama v3.3 70B | 1 | 1 | 1000 | 1000 | 2,909 output tokens/sec | 1x H200 | DGX H200 | FP8 | TensorRT-LLM 1.0 | NVIDIA H200 |
| Llama v3.3 70B | 1 | 1 | 1000 | 2000 | 2,994 output tokens/sec | 1x H200 | DGX H200 | FP8 | TensorRT-LLM 1.0 | NVIDIA H200 |
| Llama v3.3 70B | 1 | 1 | 2048 | 2048 | 2,003 output tokens/sec | 1x H200 | DGX H200 | FP8 | TensorRT-LLM 1.0 | NVIDIA H200 |
| Llama v3.3 70B | 1 | 1 | 20000 | 2000 | 283 output tokens/sec | 1x H200 | DGX H200 | FP8 | TensorRT-LLM 1.0 | NVIDIA H200 |

*Source: [NVIDIA Deep Learning Performance](https://developer.nvidia.com/deep-learning-performance-training-inference/ai-inference)*

---

## 3. Direct Comparison: MI300X vs H200

### Llama v3.3 70B - 1 GPU / FP8 - Performance Comparison

| Input Length | Output Length | MI300X (tok/s) | H200 (tok/s) | MI300X vs H200 |
|--------------|---------------|----------------|--------------|----------------|
| 128 | 2048 | **X,XXX** | 4,336 | **XX%** |
| 128 | 4096 | **X,XXX** | 2,872 | **XX%** |
| 2048 | 128 | **XXX** | 442 | **XX%** |
| 5000 | 500 | **XXX** | 566 | **XX%** |
| 500 | 2000 | **X,XXX** | 3,666 | **XX%** |
| 1000 | 1000 | **X,XXX** | 2,909 | **XX%** |
| 1000 | 2000 | **X,XXX** | 2,994 | **XX%** |
| 2048 | 2048 | **X,XXX** | 2,003 | **XX%** |
| 20000 | 2000 | **XXX** | 283 | **XX%** |

### Summary Statistics

| Metric | MI300X | H200 | Δ |
|--------|--------|------|---|
| Average Throughput | X,XXX tok/s | 2,230 tok/s | XX% |
| Best Case (128→2048) | X,XXX tok/s | 4,336 tok/s | XX% |
| Worst Case (20000→2000) | XXX tok/s | 283 tok/s | XX% |
| GPU Memory | 192 GB | 141 GB | +36% |
| Memory Bandwidth | 5.3 TB/s | 4.8 TB/s | +10% |

---

## 4. Test Configuration Details

### 4.1 AMD MI300X Test Environment

| Parameter | Value |
|-----------|-------|
| GPU | AMD Instinct MI300X |
| GPU Memory | 192 GB HBM3 |
| Memory Bandwidth | 5.3 TB/s |
| Platform | DigitalOcean Gradient AI (AMD Developer Cloud) |
| ROCm Version | 7.0.0 |
| Framework | vLLM 0.9.2 |
| Model | meta-llama/Llama-3.3-70B-Instruct |
| Quantization | FP8 |
| Max Model Length | 32,768 tokens |
| GPU Memory Utilization | 90% |

### 4.2 NVIDIA H200 Reference Environment

| Parameter | Value |
|-----------|-------|
| GPU | NVIDIA H200 141GB |
| GPU Memory | 141 GB HBM3e |
| Memory Bandwidth | 4.8 TB/s |
| Platform | DGX H200 |
| Framework | TensorRT-LLM 1.0 |
| Model | Llama v3.3 70B |
| Quantization | FP8 |

### 4.3 Benchmark Methodology

- **Throughput Calculation:** `output tokens/sec = total generated tokens / total latency`
- **Time to First Token:** Included in total latency (same as NVIDIA methodology)
- **Batch Size:** 1 (single request, max throughput)
- **Warmup:** 3 iterations before measurement
- **Measurement:** Average of 5 runs per configuration

---

## 5. Validation: Reproducing AMD Published Results

### 5.1 AMD Official Claims vs Our Results

[If AMD has published MI300X benchmarks for Llama 70B, include comparison here]

| AMD Published | Our Result | Variance |
|---------------|------------|----------|
| [Metric] | [Value] | [%] |

### 5.2 Notes on Reproducibility

[Document any deviations or observations]

---

## 6. Tooling & Workflow

### 6.1 Tools Used

| Tool | Version | Purpose |
|------|---------|---------|
| vLLM | 0.9.2 | LLM inference engine (ROCm build) |
| ROCm | 7.0.0 | AMD GPU runtime |
| doctl | latest | DigitalOcean CLI for automation |
| rocm-smi | (ROCm) | GPU monitoring |

### 6.2 Benchmark Script Usage

```bash
# Run full benchmark suite matching NVIDIA test matrix
./runara-benchmark.sh --model llama-3.3-70b --precision fp8

# Output: results/benchmark_nvidia_format.json
```

### 6.3 Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| vLLM FP8 support on ROCm | Used ROCm-optimized vLLM build with FP8 quantization |
| Matching NVIDIA methodology | Implemented identical input/output length matrix |
| Long context (20k tokens) | Required memory optimization flags |

---

## 7. Deliverables

### Included Artifacts

| File | Description |
|------|-------------|
| `benchmark_nvidia_format.json` | Raw results in NVIDIA-compatible JSON |
| `benchmark_nvidia_format.csv` | Results in CSV for easy comparison |
| `runara-benchmark.sh` | Main orchestration script |
| `benchmark-suite.py` | Python benchmark runner |
| `cleanup-runara.sh` | Emergency instance cleanup |
| `gpu_metrics.log` | GPU utilization during benchmark |

### JSON Output Format (NVIDIA-Compatible)

```json
{
  "model": "Llama v3.3 70B",
  "gpu": "1x MI300X",
  "server": "DO Gradient",
  "precision": "FP8",
  "framework": "vLLM 0.9.2",
  "gpu_version": "AMD MI300X",
  "benchmarks": [
    {
      "pp": 1,
      "tp": 1,
      "input_length": 128,
      "output_length": 2048,
      "throughput_tokens_per_sec": XXXX
    },
    {
      "pp": 1,
      "tp": 1,
      "input_length": 128,
      "output_length": 4096,
      "throughput_tokens_per_sec": XXXX
    }
    // ... all 9 test configurations
  ]
}
```

---

## 8. Conclusions

### Key Findings

1. **Llama-3.3-70B FP8 runs successfully on 1x AMD MI300X** with XX% of H200 performance
2. **Memory advantage:** MI300X (192GB) provides 36% more VRAM than H200 (141GB)
3. **Bandwidth advantage:** MI300X (5.3 TB/s) provides 10% more bandwidth than H200 (4.8 TB/s)
4. **Best scenario:** [Input→Output] achieved XX% of H200 throughput
5. **Worst scenario:** [Input→Output] achieved XX% of H200 throughput

### Recommendations

[Based on benchmark results]

---

## Appendix A: Raw Benchmark Data

[Full JSON output]

## Appendix B: GPU Diagnostics

```
[rocm-smi output during benchmarks]
```

## Appendix C: Reproducibility Notes

[Detailed steps to reproduce these results]

---

**Document Version:** 1.0  
**Benchmark Date:** [DATE]  
**Confidentiality:** Runara Proprietary
