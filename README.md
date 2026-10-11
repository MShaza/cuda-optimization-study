# cuda-optimization-study
# CUDA Optimization Study

Parallel reduction and matrix multiplication in CUDA C++, benchmarked
and analyzed against a CPU baseline and cuBLAS.

## Environment
- GPU: NVIDIA Tesla T4 (15 GB), Google Colab
- CUDA toolkit 13.0, driver 580.82.07
- Compiled with: nvcc -O3 -arch=sm_75
- Timing: cudaEvent on the GPU, warm-up run excluded, averaged over repeats

## Build and run
nvcc -O3 -arch=sm_75 reduction.cu -o reduction && ./reduction
nvcc -O3 -arch=sm_75 matmul.cu -o matmul -lcublas && ./matmul

## 1. Parallel reduction (16M floats)
| Version   | Time (ms) | Bandwidth (GB/s) |
|-----------|-----------|------------------|
| Naive     | 2.501     | 26.8             |
| Optimized | 0.424     | 158.3            |

T4 peak memory bandwidth is about 320 GB/s. Reduction is memory-bound
(one add per float loaded), so the goal is to approach that limit.

- Naive: interleaved addressing (tid % (2*stride)) causes warp divergence.
- Optimized: two loads per thread, warp-shuffle reduction (no shared-memory
  traffic or __syncthreads inside a warp), coalesced access.

## 2. Matrix multiplication (FP32, row-major)
| N    | Impl     | Time (ms) | GFLOPS | Max diff vs cuBLAS |
|------|----------|-----------|--------|--------------------|
| 512  | cuBLAS   | 0.124     | 2170.3 | 0         |
| 512  | naive    | 0.803     | 334.4  | 0         |
| 512  | tiled    | 0.643     | 417.4  | 0         |
| 512  | cpu-1thr | 35.297    | 7.6    | 3.8e-05   |
| 1024 | cuBLAS   | 0.755     | 2844.5 | 0         |
| 1024 | naive    | 6.155     | 348.9  | 6.7e-04   |
| 1024 | tiled    | 4.835     | 444.2  | 6.7e-04   |
| 1024 | cpu-1thr | 273.312   | 7.9    | 6.7e-04   |
| 2048 | cuBLAS   | 5.473     | 3139.1 | 0         |
| 2048 | naive    | 36.169    | 475.0  | 1.8e-03   |
| 2048 | tiled    | 23.601    | 727.9  | 1.8e-03   |

Key results:
- Tiled GPU vs single-threaded CPU (N=1024): about 56x faster
- Tiled vs naive (N=2048): 1.53x
- Tiled reaches about 23% of cuBLAS throughput (N=2048)

## 3. Analysis
- Correctness: differences from cuBLAS (about 1e-6 relative) come from
  floating-point addition order, not bugs. Hand-written versions sum in
  the same order, which is why their diffs match.
- Why tiling helps: each global-memory value is reused 32 times from
  shared memory instead of once, cutting global reads by about 32x.
- Why the gain is only 1.5x: the naive kernel's reads are already
  coalesced, and L1/L2 caches absorb repeated traffic. The tiled kernel
  is then limited by shared-memory reads (two per multiply-add).
- Why cuBLAS is faster: register tiling, vectorized loads, and
  double buffering, which this project does not implement.
- Profiling: [write what you actually ran and measured, or state that
  hardware counters were unavailable on Colab and that analysis is based
  on measured throughput versus hardware peak.]

## 4. What I would try next
Register tiling (each thread computes a small block of outputs),
vectorized float4 loads, double buffering, and Tensor Core (WMMA)
kernels with mixed precision.