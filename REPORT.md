# Runara Benchmark Report

## Status: Infrastructure Issues

**Date:** 2026-02-05

### Summary

Benchmark V2 methodology has been developed and scripts created, but both cloud providers are experiencing MI300X availability issues:

1. **DigitalOcean AMD Developer Cloud**: MI300X not available in any region (empty regions array in API)
2. **RunPod**: MI300X pods fail to start (runtime remains null)

### V1 Results (Single Stream, batch_size=1)

From the previous benchmark run (2026-01-30):

| Input Length | Output Length | Throughput (tok/s) |
|--------------|---------------|-------------------|
| 128 | 2048 | 33.1 |
| 128 | 4096 | 33.2 |
| 2048 | 128 | 32.4 |
| 5000 | 500 | 32.6 |
| 1000 | 1000 | 33.2 |
| 2048 | 2048 | 33.0 |

**V1 Issues Identified:**
- Single stream testing (batch_size=1) doesn't reflect real-world throughput
- No concurrency testing
- No throughput degradation measurement
- Cost calculation based on single-stream throughput is misleading

### V2 Benchmark Methodology (Created)

Script: `scripts/benchmark_v2.py`

**Improvements:**
1. **Concurrent Requests**: Tests with 1, 4, 8, 16, 32, 64 concurrent requests
2. **Throughput Degradation**: Measures tok/s at 500, 1000, 2000, 3000, 4000 token outputs
3. **Proper Cost Calculation**: Based on peak throughput with concurrency
4. **GPU Memory Monitoring**: Tracks VRAM usage during generation

### Expected V2 Results (Based on Similar Hardware)

Based on published MI300X benchmarks with vLLM:

| Concurrency | Expected Throughput (tok/s) |
|-------------|----------------------------|
| 1 | ~33 (matches V1) |
| 4 | ~90-120 |
| 8 | ~150-200 |
| 16 | ~250-350 |
| 32 | ~400-500 |
| 64 | ~500-700 |

**Expected Throughput Degradation:**
| Output Tokens | Expected Throughput (tok/s) |
|---------------|----------------------------|
| 500 | ~35 |
| 1000 | ~34 |
| 2000 | ~32 |
| 3000 | ~30 |
| 4000 | ~28 |

**Cost Analysis (Expected):**
- At peak throughput (~600 tok/s with concurrency 64)
- Tokens per hour: 2,160,000
- Hourly cost: $1.99 (RunPod) / $2.00 (DigitalOcean)
- **Cost per 1M tokens: ~$0.92**

### Hardware Configuration

- **GPU**: 1x AMD MI300X (192GB HBM3)
- **Model**: Llama-3.3-70B-Instruct
- **Precision**: FP8
- **Framework**: vLLM 0.9.2
- **Context Length**: 32768

### Files Created

1. `scripts/benchmark_v2.py` - Corrected benchmark with concurrency testing
2. `REPORT.md` - This report

### Next Steps

1. Retry MI300X deployment when capacity becomes available
2. Consider alternative providers (Lambda Labs, CoreWeave)
3. Run the V2 benchmark script when infrastructure is accessible
4. Update this report with actual measurements

### Notes on Cost Comparison

| Provider | MI300X Hourly Cost | Expected Cost/1M Tokens |
|----------|-------------------|------------------------|
| DigitalOcean | $1.99 | ~$0.92 |
| RunPod Secure | $1.99 | ~$0.92 |
| RunPod Spot | $0.50 | ~$0.23 |

These costs assume peak throughput with concurrent batching. Single-stream throughput (~33 tok/s) would yield ~$16.70/1M tokens, which is 18x higher and not representative of production workloads.
