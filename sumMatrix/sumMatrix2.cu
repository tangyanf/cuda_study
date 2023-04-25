#include "freshman.hpp"
#include <cuda_runtime.h>

__global__ void sumMatrix(float* MatA, float* MatB, float* MatC, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * nx;
    MatC[idx] = MatA[idx] + MatB[idx];
}

int main(int argc, char** argv) {
    initDevice(0);
    int nx = 1 << 13;
    int ny = 1 << 13;
    int size = nx * ny;
    int nBytes = size * sizeof(float);

    //malloc
    float* host_a = (float*)malloc(nBytes);
    float* host_b = (float*)malloc(nBytes);
    float* host_c = (float*)malloc(nBytes);
    float* c_from_dev = (float*)malloc(nBytes);
    initData(host_a, size);
    initData(host_b, size);

    // dev malloc
    float* dev_a = NULL;
    float* dev_b = NULL;
    float* dev_c = NULL;
    CHECK(cudaMalloc((void**)&dev_a, nBytes));
    CHECK(cudaMalloc((void**)&dev_b, nBytes));
    CHECK(cudaMalloc((void**)&dev_c, nBytes));

    CHECK(cudaMemcpy(dev_a, host_a, nBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(dev_b, host_b, nBytes, cudaMemcpyHostToDevice));

    int dimx = argc>2?atoi(argv[1]):32;
    int dimy = argc>2?atoi(argv[2]):32;

    dim3 block(dimx,dimy);
    dim3 grid((nx-1)/block.x+1,(ny-1)/block.y+1);
    double iStart = cpuSecond();
    sumMatrix<<<grid, block>>>(dev_a, dev_b, dev_c, nx, ny);
    double iElaps = cpuSecond() - iStart;
    printf("GPU execution configuration<<<(%d,%d),(%d,%d)>>> time elapsed: %lf sec\n",grid.x,grid.y,block.x,block.y,iElaps);
    CHECK(cudaMemcpy(c_from_dev, dev_c, nBytes, cudaMemcpyDeviceToHost));

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(host_a);
    free(host_b);
    free(host_c);
    free(c_from_dev);
    return 0;
}