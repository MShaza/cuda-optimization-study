#include <cstdio>

__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  int n = 1 << 20;                  // 1,048,576 elements
  size_t sz = n * sizeof(float);
  float *a, *b, *c;
  cudaMallocManaged(&a, sz);
  cudaMallocManaged(&b, sz);
  cudaMallocManaged(&c, sz);
  for (int i = 0; i < n; i++) { a[i] = 1.0f; b[i] = 2.0f; }

  add<<<(n + 255) / 256, 256>>>(a, b, c, n);
  cudaDeviceSynchronize();

  printf("c[0]=%f c[n-1]=%f (expect 3)\n", c[0], c[n - 1]);
  cudaFree(a); cudaFree(b); cudaFree(c);
}
