#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#define CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error: %s (line %d)\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)

// V1: naive. Shared memory, interleaved addressing (causes warp divergence)
__global__ void reduceNaive(const float* in, float* out, int n){
  extern __shared__ float s[];
  int tid = threadIdx.x, i = blockIdx.x * blockDim.x + tid;
  s[tid] = (i < n) ? in[i] : 0.f;
  __syncthreads();
  for (int stride = 1; stride < blockDim.x; stride *= 2) {
    if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];
    __syncthreads();
  }
  if (tid == 0) out[blockIdx.x] = s[0];
}

// V2: optimized. 2 loads per thread, warp shuffle, minimal shared memory
__global__ void reduceOpt(const float* in, float* out, int n){
  __shared__ float s[32];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x * 2 + tid;
  float v = 0.f;
  if (i < n) v += in[i];
  if (i + blockDim.x < n) v += in[i + blockDim.x];
  for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
  int lane = tid & 31, warp = tid >> 5;
  if (lane == 0) s[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (tid < blockDim.x / 32) ? s[lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffff, v, off);
    if (tid == 0) out[blockIdx.x] = v;
  }
}

int main(){
  const int n = 1 << 24, T = 256;
  std::vector<float> h(n, 1.0f);                 // expected sum = n
  float *d_in, *d_out;
  CHECK(cudaMalloc(&d_in, n * sizeof(float)));
  CHECK(cudaMalloc(&d_out, (n / T + 1) * sizeof(float)));
  CHECK(cudaMemcpy(d_in, h.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  std::vector<float> partial(n / T + 1);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);

  for (int version = 0; version < 2; version++) {
    int blocks = version == 0 ? (n + T - 1) / T : (n + 2 * T - 1) / (2 * T);
    float ms = 0; double sum = 0;
    for (int r = 0; r < 6; r++) {                // r==0 is warm-up
      cudaEventRecord(a);
      if (version == 0) reduceNaive<<<blocks, T, T * sizeof(float)>>>(d_in, d_out, n);
      else              reduceOpt<<<blocks, T>>>(d_in, d_out, n);
      cudaEventRecord(b); cudaEventSynchronize(b);
      float t; cudaEventElapsedTime(&t, a, b);
      if (r > 0) ms += t;
    }
    ms /= 5;
    CHECK(cudaMemcpy(partial.data(), d_out, blocks * sizeof(float), cudaMemcpyDeviceToHost));
    for (int i = 0; i < blocks; i++) sum += partial[i];
    double gbs = (double)n * sizeof(float) / (ms * 1e-3) / 1e9;
    printf("%-10s time=%.3f ms  bandwidth=%.1f GB/s  sum=%.0f (expect %d)\n",
           version == 0 ? "naive" : "optimized", ms, gbs, sum, n);
  }
}
