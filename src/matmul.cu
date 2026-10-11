#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <chrono>
#include <functional>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define CHECK(x) do{cudaError_t e=(x); if(e!=cudaSuccess){printf("CUDA error: %s (line %d)\n",cudaGetErrorString(e),__LINE__);exit(1);}}while(0)
#define TILE 32

void cpuMatmul(const float*A,const float*B,float*C,int N){
  for(int i=0;i<N*N;i++) C[i]=0;
  for(int i=0;i<N;i++) for(int k=0;k<N;k++){
    float a=A[i*N+k];
    for(int j=0;j<N;j++) C[i*N+j]+=a*B[k*N+j];
  }
}

__global__ void mmNaive(const float*A,const float*B,float*C,int N){
  int row=blockIdx.y*blockDim.y+threadIdx.y, col=blockIdx.x*blockDim.x+threadIdx.x;
  if(row<N && col<N){
    float s=0;
    for(int k=0;k<N;k++) s+=A[row*N+k]*B[k*N+col];
    C[row*N+col]=s;
  }
}

__global__ void mmTiled(const float*A,const float*B,float*C,int N){
  __shared__ float As[TILE][TILE], Bs[TILE][TILE];
  int tx=threadIdx.x, ty=threadIdx.y;
  int row=blockIdx.y*TILE+ty, col=blockIdx.x*TILE+tx;
  float s=0;
  for(int t=0;t<(N+TILE-1)/TILE;t++){
    int ac=t*TILE+tx, br=t*TILE+ty;
    As[ty][tx]=(row<N && ac<N)? A[row*N+ac] : 0.f;
    Bs[ty][tx]=(br<N && col<N)? B[br*N+col] : 0.f;
    __syncthreads();
    for(int k=0;k<TILE;k++) s+=As[ty][k]*Bs[k][tx];
    __syncthreads();
  }
  if(row<N && col<N) C[row*N+col]=s;
}

float timeGpu(std::function<void()> f,int reps=5){
  cudaEvent_t a,b; cudaEventCreate(&a); cudaEventCreate(&b);
  f(); cudaDeviceSynchronize();                 // warm-up
  cudaEventRecord(a);
  for(int i=0;i<reps;i++) f();
  cudaEventRecord(b); cudaEventSynchronize(b);
  float ms; cudaEventElapsedTime(&ms,a,b);
  return ms/reps;
}

int main(){
  cublasHandle_t h; cublasCreate(&h);
  printf("%6s | %-9s | %10s | %9s | %s\n","N","impl","time(ms)","GFLOPS","max|diff| vs cuBLAS");
  for(int N : {512,1024,2048}){
    size_t bytes=(size_t)N*N*sizeof(float);
    std::vector<float> hA(N*N),hB(N*N),hC(N*N),ref(N*N);
    for(int i=0;i<N*N;i++){hA[i]=(rand()%100)/100.f; hB[i]=(rand()%100)/100.f;}
    float *A,*B,*C; CHECK(cudaMalloc(&A,bytes)); CHECK(cudaMalloc(&B,bytes)); CHECK(cudaMalloc(&C,bytes));
    CHECK(cudaMemcpy(A,hA.data(),bytes,cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(B,hB.data(),bytes,cudaMemcpyHostToDevice));
    dim3 blk(TILE,TILE), grd((N+TILE-1)/TILE,(N+TILE-1)/TILE);
    double flop=2.0*N*N*N;
    auto report=[&](const char*name,float ms,const std::vector<float>&r){
      double diff=0;
      if(!r.empty()) for(int i=0;i<N*N;i++) diff=fmax(diff,fabs(hC[i]-r[i]));
      printf("%6d | %-9s | %10.3f | %9.1f | %.4g\n",N,name,ms,flop/(ms*1e6),diff);
    };

    float one=1,zero=0;
    auto cub=[&](){cublasSgemm(h,CUBLAS_OP_N,CUBLAS_OP_N,N,N,N,&one,B,N,A,N,&zero,C,N);};
    float msC=timeGpu(cub);
    CHECK(cudaMemcpy(ref.data(),C,bytes,cudaMemcpyDeviceToHost));
    hC=ref; report("cuBLAS",msC,{});

    float msN=timeGpu([&](){mmNaive<<<grd,blk>>>(A,B,C,N);});
    CHECK(cudaMemcpy(hC.data(),C,bytes,cudaMemcpyDeviceToHost)); report("naive",msN,ref);

    float msT=timeGpu([&](){mmTiled<<<grd,blk>>>(A,B,C,N);});
    CHECK(cudaMemcpy(hC.data(),C,bytes,cudaMemcpyDeviceToHost)); report("tiled",msT,ref);

    if(N<=1024){                                // CPU is too slow beyond this
      auto t0=std::chrono::steady_clock::now();
      cpuMatmul(hA.data(),hB.data(),hC.data(),N);
      float msCpu=std::chrono::duration<float,std::milli>(std::chrono::steady_clock::now()-t0).count();
      report("cpu-1thr",msCpu,ref);
    }
    cudaFree(A); cudaFree(B); cudaFree(C);
  }
}
