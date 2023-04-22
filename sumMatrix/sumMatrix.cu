#include "freshman.hpp"
#include <cuda_runtime.h>

void sumMatrix2D_cpu(float* MatA, float* MatB, float* MatC, int nx, int ny) {
    float* a = MatA;
    float* b = MatB;
    float* c = MatC;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            c[j] = a[j] + b[j];
        }
        a += nx;
        b += nx;
        c += nx;
    }
}

__global__ void sumMatrix(float* MatA, float* MatB, float* MatC, int nx, int ny) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x;
    int iy = threadIdx.y + blockDim.y * blockIdx.y;
    int idx = ix + iy * nx;
    MatC[idx] = MatA[idx] + MatB[idx];
}

int main() {
    initDevice(0);
    int nx = 1 << 12;
    int ny = 1 << 12;
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

    int dimx = 32;
    int dimy = 32;

    double iStart = cpuSecond();
    sumMatrix2D_cpu(host_a, host_b, host_c, nx, ny);
    double iElaps = cpuSecond() - iStart;
    printf("Cpu Execution time elapsed: %lf sec\n", iElaps);

    // 2d block and 2d grid
    dim3 block_0(dimx, dimy);
    dim3 grid_0((nx-1)/block_0.x+1, (ny-1)/block_0.y+1);
    iStart = cpuSecond();
    sumMatrix<<<grid_0, block_0>>>(dev_a, dev_b, dev_c, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("GPU execution configuration<<<(%d, %d),(%d,%d)>>> time elapsed: %lf sec\n",grid_0.x,grid_0.y,block_0.x, block_0.y, iElaps);
    CHECK(cudaMemcpy(c_from_dev, dev_c, nBytes, cudaMemcpyDeviceToHost));
    checkResult(host_c, c_from_dev, size);

    // 1d block and 1d grid
    dim3 block_1(dimx);
    dim3 grid_1((size - 1)/block_1.x + 1);
    iStart = cpuSecond();
    sumMatrix<<<grid_1, block_1>>>(dev_a, dev_b, dev_c, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("GPU execution configuration<<<(%d),(%d)>>> time elapsed: %lf sec\n",grid_1.x,block_1.x,iElaps);
    CHECK(cudaMemcpy(c_from_dev, dev_c, nBytes, cudaMemcpyDeviceToHost));
    checkResult(host_c, c_from_dev, size);

    // 2d block and 1d grid
    dim3 block_2(dimx, dimy);
    dim3 grid_2((size - 1)/(dimx*dimy) +1);
    iStart = cpuSecond();
    sumMatrix<<<grid_2, block_2>>>(dev_a, dev_b, dev_c, nx, ny);
    iElaps = cpuSecond() - iStart;
    printf("GPU execution configuration<<<(%d),(%d, %d)>>> time elapsed: %lf sec\n",grid_2.x,block_2.x,block_2.y,iElaps);
    CHECK(cudaMemcpy(c_from_dev, dev_c, nBytes, cudaMemcpyDeviceToHost));
    checkResult(host_c, c_from_dev, size);
    
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    free(host_a);
    free(host_b);
    free(host_c);
    free(c_from_dev);
    return 0;
}