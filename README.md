Scientific benchmarking framework for CUDA matrix multiplication on NVIDIA Jetson Orin Nano. Four implementations, three power modes, five matrix sizes. Every result mathematically validated to 99.5% accuracy.

<p align="center">
  <a href="https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis" target="_blank">
    <img src="https://raw.githubusercontent.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/main/docs/assets/jetson_orin_nano_matmul_power_benchmarks_latest_v1.0.0.png" alt="Jetson Orin Nano Power-Performance Benchmarks v1.0.0" width="640"/>
  </a>
</p>

<p align="center">
  <a href="https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/releases/tag/v1.0.0">
    <img src="https://img.shields.io/badge/version-v1.0.0-brightgreen" alt="Version: v1.0.0">
  </a>
  <a href="https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis/blob/main/LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT">
  </a>
  <a href="https://orcid.org/0009-0006-0322-7974">
    <img src="https://img.shields.io/badge/ORCID-0009--0006--0322--7974-A6CE39?logo=orcid&logoColor=white" alt="ORCID: 0009-0006-0322-7974">
  </a>
  <a href="https://bytestacklabs.com">
    <img src="https://img.shields.io/badge/Made%20by-ByteStack%20Labs-2ea44f" alt="ByteStack Labs">
  </a>
</p>

---

## What This Measures

Matrix multiplication is the computational primitive underlying neural network inference, training, and linear algebra workloads. This project benchmarks four distinct CUDA implementations across the Jetson Orin Nano's three power modes (15W, 25W, MAXN) to quantify the relationship between power budget, implementation strategy, and throughput.

The four implementations span the optimization spectrum: naive element-wise computation, cache-blocked tiling for memory hierarchy exploitation, cuBLAS library calls leveraging vendor-optimized kernels, and Tensor Core WMMA (Warp Matrix Multiply-Accumulate) operations using mixed-precision SM 8.7 hardware.

## Key Findings

**Peak performance:** cuBLAS reaches 1,282 GFLOPS (61% of theoretical peak at 2,089 GFLOPS @ 1020 MHz). Tensor Cores (TF32) reach 952 GFLOPS, delivering 10.0x speedup over naive at 1024x1024 with a max error of 0.00972 versus less than 1e-6 for FP32.

**Power efficiency:** 25W mode achieves 90% of MAXN performance at 88% power consumption. At 25W, cuBLAS delivers 1,150 GFLOPS at 52 GFLOPS/W. MAXN delivers 1,282 GFLOPS at 51 GFLOPS/W. The 25W mode is the optimal performance-per-watt configuration.

**Validation:** 60 validated data points across all implementations, power modes, and matrix sizes. 99.5% measurement accuracy. Numerical accuracy below 1e-5 (FP32) and below 0.01 (TF32). All metrics validated against theoretical hardware limits with real-time GPU frequency measurement.

| Implementation | Peak GFLOPS | Efficiency | Speedup vs Naive | Best Use Case |
| --- | --- | --- | --- | --- |
| Naive | 95 | 4.6% | 1.0x | Educational baseline |
| Blocked | 150 | 7.2% | 1.6x | Cache optimization study |
| cuBLAS | 1,282 | 61.4% | 13.5x | Production workloads |
| Tensor Core | 952 | 45.6% | 10.0x | ML inference (TF32) |

Speedup calculated at 1024x1024. Efficiency relative to theoretical peak (2,089 GFLOPS @ 1020 MHz MAXN).

## Technical Validation

Every benchmark result is validated against a reference NumPy computation. The validation pipeline computes element-wise relative error, enforces a 99.5% accuracy threshold, and flags any result that falls below it. This is not optional; the validation runs as part of the benchmark suite, not as a separate step.

**Tooling quality:** Python (Ruff, mypy, pytest), C++ (clang-format, cpplint), Shell (shellcheck). Google-style docstrings throughout. CI enforces all linters on every push.

## Quick Start

### Prerequisites

- NVIDIA Jetson Orin Nano with JetPack 6.x
- CUDA Toolkit 12.6+
- Python 3.10+
- C++14 compiler with CUDA support

### Installation

```bash
git clone https://github.com/Cre4T3Tiv3/jetson-orin-matmul-analysis.git
cd jetson-orin-matmul-analysis
make quick-start
```

### Run Benchmarks

```bash
# Quick functionality test (single power mode)
make test-quick

# Full power mode analysis (~15 minutes)
sudo make full-analysis

# Generate visualizations
make visualize
```

## Project Structure

```
cuda/
  kernels/              # CUDA implementations (naive, blocked, cuBLAS, Tensor Core)
  utils/                # Shared utilities, performance monitoring, logging
benchmarks/             # Benchmark orchestration
data/
  raw/power_modes/      # Benchmark results (JSON/CSV)
  reports/              # Analysis markdown reports
  plots/                # Visualizations
tests/                  # Pytest suite
scripts/                # Power mode configuration and automation
```

## Documentation

- [Contributing](CONTRIBUTING.md) — Development setup, coding standards, and linter configuration
- [Sudo Setup Guide](SUDO_SETUP_GUIDE.md) — Passwordless sudo for uninterrupted benchmarking

## License

[MIT](LICENSE)
