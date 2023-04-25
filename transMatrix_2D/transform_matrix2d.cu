#include "freshman.hpp"
#include <cuda_runtime.h>

void transform_matrix2d_cpu(float* a, float* b, int nx, int ny) {
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            b[i*ny+j] = a[j*nx+i];
        }
    }
}

__global__ void copyRow(float* a, float* b, int nx, int ny) {
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int idx = row * nx + col;
    if (col < nx && row < ny) {
        b[idx] = a[idx];
    }
}

__global__ void copyCol(float* a, float* b, int nx, int ny) {
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int idx = col * ny + row;
    if (col < nx && row < ny) {
        b[idx] = a[idx];
    }
}

__global__ void transformNaiveRow(float* a, float* b, int nx, int ny) {
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (col < nx && row < ny) {
        b[col * ny + row] = a[row * nx + col];
    }
}

__global__ void transformNaiveCol(float* a, float* b, int nx, int ny) {
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (col < nx && row < ny) {
        b[row * nx + col] = a[col * ny + row];
    }
}

__global__ void transformRowUnroll(float* a, float* b, int nx, int ny) {
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x * 4;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (col < nx && row < ny) {
        b[col*ny+row] = a[row*nx+col];
        b[(col+blockDim.x)*ny+row] = a[row*nx+col+blockDim.x];
        b[(col+blockDim.x*2)*ny+row] = a[row*nx+col+blockDim.x*2];
        b[(col+blockDim.x*3)*ny+row] = a[row*nx+col+blockDim.x*3];
    }
}

__global__ void transformColUnroll(float* a, float* b, int nx, int ny) {
    unsigned int col = threadIdx.x + blockDim.x * blockIdx.x*4;
    unsigned int row = threadIdx.y + blockDim.y * blockIdx.y;
    if (col < nx && row < ny) {
        b[row * nx + col] = a[col * ny + row];
        b[row * nx + col + blockDim.x] = a[(col + blockDim.y) * ny + row];
        b[row * nx + col + blockDim.x*2] = a[(col + blockDim.x*2) * ny + row];
        b[row * nx + col + blockDim.x*3] = a[(col + blockDim.x*3) * ny + row];
    }
}

__global__ void transformSmem1(float * in,float* out,int nx,int ny)
{
	__shared__ float tile[32][32];
	unsigned int ix,iy,transform_in_idx,transform_out_idx;
	ix=threadIdx.x+blockDim.x*blockIdx.x;
    iy=threadIdx.y+blockDim.y*blockIdx.y;
	transform_in_idx=iy*nx+ix;

	unsigned int bidx,irow,icol;
	bidx=threadIdx.y*blockDim.x+threadIdx.x;
	irow=bidx/blockDim.y;
	icol=bidx%blockDim.y;

	ix=blockIdx.y*blockDim.y+icol;
	iy=blockIdx.x*blockDim.x+irow;

	transform_out_idx=iy*ny+ix;

	if(ix<nx&& iy<ny)
	{
		tile[threadIdx.y][threadIdx.x]=in[transform_in_idx];
		__syncthreads();
		out[transform_out_idx]=tile[icol][irow];

	}

}

__global__ void transformSmem(float* a, float* b, int nx, int ny) {
    __shared__  float smem[32][32];
    // calculate in index
    unsigned int ix = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int transform_in_idx = iy * nx + ix;

    // calculate smem index
    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx/blockDim.y;
    unsigned int icol = bidx%blockDim.y;

    // calculate out index
    ix = blockIdx.y * blockDim.y + icol;
    iy = blockIdx.x * blockDim.x + irow;
    unsigned int transform_out_idx = iy * ny + ix;

    if (ix < nx && iy < ny) {
        smem[threadIdx.y][threadIdx.x] = a[transform_in_idx];
        __syncthreads();
        b[transform_out_idx] = smem[icol][irow];
    }
}

__global__ void transformSmemUnroll(float* a, float* b, int nx, int ny) {
    __shared__ float smem[32 * 32 * 2];
    unsigned int ix = threadIdx.x + blockDim.x * 2 * blockIdx.x;
    unsigned int iy = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned int transform_in_idx = iy * nx + ix;

    unsigned int bidx = threadIdx.y * blockDim.x + threadIdx.x;
    unsigned int irow = bidx/blockDim.y;
    unsigned int icol = bidx%blockDim.x;

    unsigned int ix2 = blockIdx.y * blockDim.y + icol;
    unsigned int iy2 = blockIdx.x * (blockDim.x * 2) + irow;

    unsigned int transform_out_idx = iy2 * ny + ix2;

    if (ix + blockDim.x < nx && iy < ny) {
        unsigned int row_idx = threadIdx.y * (blockDim.x*2) + threadIdx.x;
        smem[row_idx] = a[transform_in_idx];
        smem[row_idx + blockDim.x] = a[transform_in_idx + 32];

        __syncthreads();
    }
}

int main(int argc, char** argv) {
    initDevice(0);
    int nx = 1 << 12;
    int ny = 1 << 12;
    int dimx = 32;
    int dimy = 32;
    if (argc > 1) {
        dimx = atoi(argv[1]) > 1 ? atoi(argv[1]) : dimx;
        dimy = atoi(argv[2]) > 1 ? atoi(argv[1]) : dimy;
    }
    int nxy = nx * ny;
    int nbytes = nxy * sizeof(float);

    float* host_a = (float*)malloc(nbytes);
    float* host_b = (float*)malloc(nbytes);
    float* host_c_from_dev = (float*)malloc(nbytes);
    initData(host_a, nxy);

    float* dev_a = NULL;
    float* dev_b = NULL;
    CHECK(cudaMalloc((void**)&dev_a, nbytes));
    CHECK(cudaMalloc((void**)&dev_b, nbytes));
    CHECK(cudaMemcpy(dev_a, host_a, nbytes, cudaMemcpyHostToDevice));

    double iStart = cpuSecond();
    transform_matrix2d_cpu(host_a, host_b, nx, ny);
    double iElaps = cpuSecond() - iStart;
    printf("CPU Execution time elaps %lf ms\n", iElaps);
    // printMatrix(host_a, nx, ny);
    // printMatrix(host_b, ny, nx);

    dim3 block1(dimx, dimy);
    dim3 grid1((nx - 1)/block1.x + 1, (ny - 1)/block1.y + 1);
    dim3 grid2((nx - 1)/(block1.x*4) + 1, (ny - 1)/block1.y + 1);
    dim3 grid3((nx - 1)/block1.x + 1, (ny - 1)/(block1.y*4) + 1);
    iStart = cpuSecond();
    copyRow<<<grid1, block1>>>(dev_a, dev_b, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("copyRow<<<(%d,%d), (%d,%d)>>> Execution time elaps %lf ms\n",grid1.x, grid1.y, block1.x, block1.y, iElaps);
    CHECK(cudaMemcpy(host_c_from_dev, dev_b, nbytes, cudaMemcpyDeviceToHost));
    // printMatrix(host_c_from_dev, ny, nx);

    iStart = cpuSecond();
    copyCol<<<grid1, block1>>>(dev_a, dev_b, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("copyCol<<<(%d,%d), (%d,%d)>>> Execution time elaps %lf ms\n",grid1.x, grid1.y, block1.x, block1.y, iElaps);
    CHECK(cudaMemcpy(host_c_from_dev, dev_b, nbytes, cudaMemcpyDeviceToHost));
    // printMatrix(host_c_from_dev, ny, nx);

    iStart = cpuSecond();
    transformNaiveRow<<<grid1, block1>>>(dev_a, dev_b, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("transformNaiveRow<<<(%d,%d), (%d,%d)>>> Execution time elaps %lf ms\n",grid1.x, grid1.y, block1.x, block1.y, iElaps);
    CHECK(cudaMemcpy(host_c_from_dev, dev_b, nbytes, cudaMemcpyDeviceToHost));
    // printMatrix(host_c_from_dev, ny, nx);

    iStart = cpuSecond();
    transformNaiveCol<<<grid1, block1>>>(dev_a, dev_b, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("transformNaiveCol<<<(%d,%d), (%d,%d)>>> Execution time elaps %lf ms\n",grid1.x, grid1.y, block1.x, block1.y, iElaps);
    CHECK(cudaMemcpy(host_c_from_dev, dev_b, nbytes, cudaMemcpyDeviceToHost));
    checkResult(host_c_from_dev, host_b, nxy);
    // printMatrix(host_c_from_dev, ny, nx);

    iStart = cpuSecond();
    transformRowUnroll<<<grid2, block1>>>(dev_a, dev_b, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("transformRowUnroll<<<(%d,%d), (%d,%d)>>> Execution time elaps %lf ms\n",grid2.x, grid2.y, block1.x, block1.y, iElaps);
    CHECK(cudaMemcpy(host_c_from_dev, dev_b, nbytes, cudaMemcpyDeviceToHost));

    checkResult(host_c_from_dev, host_b, nxy);

    iStart = cpuSecond();
    transformColUnroll<<<grid3, block1>>>(dev_a, dev_b, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("transformColUnroll<<<(%d,%d), (%d,%d)>>> Execution time elaps %lf ms\n",grid3.x, grid3.y, block1.x, block1.y, iElaps);
    CHECK(cudaMemcpy(host_c_from_dev, dev_b, nbytes, cudaMemcpyDeviceToHost));
    checkResult(host_c_from_dev, host_b, nxy);

    iStart = cpuSecond();
    transformSmem<<<grid1, block1>>>(dev_a, dev_b, nx, ny);
    cudaDeviceSynchronize();
    iElaps = cpuSecond() - iStart;
    printf("transformSmem<<<(%d,%d), (%d,%d)>>> Execution time elaps %lf ms\n",grid1.x, grid1.y, block1.x, block1.y, iElaps);
    CHECK(cudaMemcpy(host_c_from_dev, dev_b, nbytes, cudaMemcpyDeviceToHost));
    checkResult(host_c_from_dev, host_b, nxy);

    return 0;
}