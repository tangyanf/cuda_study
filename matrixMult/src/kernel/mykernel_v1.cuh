#include <cuda_runtime.h>

__global__ void mysgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    int gx = threadIdx.x + blockDim.x * blockIdx.x;
    int gy = threadIdx.y + blockDim.y * blockIdx.y;
    float tmp = 0;
    for (int i = 0; i < K; i++) {
        tmp += A[gy * K + i] * B[i * N + gx];
    }
    C[gy*N+gx] = alpha * tmp + beta * C[gy*N+gx];
}