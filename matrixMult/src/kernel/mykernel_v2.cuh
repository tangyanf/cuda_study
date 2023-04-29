#pragma once
#include <cuda_runtime.h>

template <const int BLOCK_SIZE>
__global__ void mysgemm_v2(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C)
{
    int bx = blockIdx.x;
    int by = blockIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;

    int tx = threadIdx.x % BN;
    int ty = threadIdx.x / BN;

    __shared__ float tileA[BM * BK];
    __shared__ float tileB[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    float tmp = 0.;
    for (int k = 0; k < K; k += BK)
    {
        tileA[ty * BK + tx] = A[ty * K + tx];
        tileB[ty * BN + tx] = B[ty * N + tx];
        __syncthreads();
        A += BK;
        B += BK * N;
        for (int i = 0; i < BK; i++)
        {
            tmp += tileA[ty * BK + i] * tileB[i * BN + tx];
        }
        __syncthreads();
    }
    C[ty * N + tx] = alpha * tmp + beta * C[ty * N + tx];
}
