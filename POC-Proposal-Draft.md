# Runara AI — 1-Week POC Proposal
**Prepared by:** Progforce  
**Date:** January 29, 2026  
**Version:** Draft for Review

---

## Executive Summary

We propose a 1-week Proof of Concept to demonstrate the feasibility of running NVIDIA-targeted AI models on AMD GPUs. The POC will use **Llama-3.3-70B FP8** as the test model, running on AMD MI300X hardware via Hot Aisle cloud.

**Goal:** Prove that existing PyTorch/CUDA models can be automatically re-targeted to AMD GPUs with acceptable performance.

---

## Scope & Deliverables

### What We Will Deliver in 1 Week

| # | Deliverable | Description |
|---|-------------|-------------|
| 1 | **Working Demo** | Llama-3.3-70B FP8 running on AMD MI300X with inference capability |
| 2 | **Benchmark Comparison** | Side-by-side performance metrics (tokens/sec, latency, throughput) on AMD vs NVIDIA |
| 3 | **Tooling Documentation** | Report on tools used (ROCm, PyTorch+ROCm, vLLM, etc.) with setup guide |
| 4 | **Challenges Report** | Document any issues encountered and solutions/workarounds |
| 5 | **Reproducible Workflow** | Step-by-step guide for Runara to replicate the setup |

### What's Out of Scope (for this POC)

- Full automation pipeline (Phase 2+)
- Custom tooling development
- MLIR/compiler-level work (exploration only)
- Multi-model testing beyond Llama-3.3-70B

---

## Technical Approach

### Phase 1: Environment Setup (Day 1-2)
- Provision AMD MI300X instance on Hot Aisle
- Provision NVIDIA H100/A100 baseline on RunPod
- Install ROCm stack, PyTorch with ROCm support
- Validate basic GPU functionality

### Phase 2: Model Deployment (Day 2-3)
- Deploy Llama-3.3-70B FP8 on AMD using:
  - vLLM with ROCm backend (preferred path)
  - Or: Hugging Face Transformers + PyTorch-ROCm
- Run initial inference tests
- Debug and resolve compatibility issues

### Phase 3: Benchmarking (Day 4-5)
- Run standardized benchmarks on both platforms:
  - Tokens per second (generation)
  - Time to first token (latency)
  - Throughput at various batch sizes
  - Memory utilization
- Format results for direct NVIDIA comparison

### Phase 4: Documentation & Delivery (Day 5-6)
- Compile tooling documentation
- Write challenges/solutions report
- Create reproducible workflow guide
- Package deliverables

### Buffer (Day 7)
- Address any remaining issues
- Final review and handoff

---

## Tools & Technologies

| Category | Tool | Purpose |
|----------|------|---------|
| AMD Software | ROCm 6.x | AMD GPU compute stack |
| ML Framework | PyTorch + ROCm | Model execution |
| Inference | vLLM (ROCm) | High-performance LLM serving |
| Benchmarking | Custom scripts / llm-benchmark | Performance measurement |
| Comparison | NVIDIA baseline | H100/A100 reference numbers |

**Exploration (time permitting):**
- rocMLIR, IREE for compiler-level insights
- AMD's official benchmark reproduction

---

## Infrastructure

### AMD (Primary Test Environment)
**Provider:** Hot Aisle  
**Hardware:** 8x AMD MI300X (192GB each = 1.5TB total VRAM)  
**Cost:** $1.99/GPU/hr × 8 = ~$15.92/hr  
**Access:** Instant self-service

### NVIDIA (Baseline Comparison)
**Provider:** RunPod  
**Hardware:** 2x H100 80GB or 1x H200 141GB  
**Cost:** ~$4-6/hr  
**Access:** Instant self-service

### Estimated Compute Costs
| Item | Hours | Cost |
|------|-------|------|
| AMD MI300X (8x) | 40 | ~$640 |
| NVIDIA H100 (2x) | 20 | ~$170 |
| **Total** | | **~$810** |

*Actual costs may vary based on debugging time and reruns.*

---

## Timeline

| Day | Activities |
|-----|------------|
| **1** | Environment setup, ROCm installation, GPU validation |
| **2** | Model download, initial deployment attempts |
| **3** | Debug compatibility issues, successful inference |
| **4** | Benchmark runs on AMD |
| **5** | NVIDIA baseline benchmarks, comparison analysis |
| **6** | Documentation, workflow guide |
| **7** | Buffer / final delivery |

**Start Date:** Upon GPU access confirmation  
**End Date:** 7 calendar days from start

---

## Pricing

| Item | Cost |
|------|------|
| Engineering (1 week) | $X,XXX |
| Cloud Compute (estimated) | ~$810 |
| **Total** | **$X,XXX** |

*[Itzik to fill in engineering rate]*

---

## Requirements from Runara

1. **GPU Access:** Sign up and provide credentials for:
   - Hot Aisle (AMD): https://hotaisle.xyz — run `ssh admin.hotaisle.app`
   - RunPod (NVIDIA): https://runpod.io

2. **Model Access:** Confirm Llama-3.3-70B FP8 weights access (Hugging Face or direct)

3. **Point of Contact:** Technical contact for questions during POC

4. **Benchmark Reference:** Any specific AMD benchmark results you'd like us to reproduce

---

## Success Criteria

The POC will be considered successful if:

1. ✅ Llama-3.3-70B FP8 runs inference on AMD MI300X
2. ✅ Benchmark results are collected and formatted for comparison
3. ✅ Documentation enables Runara to reproduce the setup
4. ✅ Clear path forward identified for automation (Phase 2)

---

## Next Steps

1. Runara reviews and approves proposal
2. Runara sets up cloud provider accounts
3. Progforce receives access credentials
4. POC begins

---

**Questions?** Contact: itzik@progforce.com
