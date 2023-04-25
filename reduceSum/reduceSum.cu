#include "freshman.hpp"
#include <cuda_runtime.h>

int recursiveReduce(int* data, int const size) {
    if (size == 1) return data[0];
    int const stride = size/2;
    if (size%2==1) {
        for (int i = 0; i < stride; i++) {
            data[i] += data[i+stride];
        }
        data[0] += data[size-1];
    } else {
        for (int i = 0; i < stride; i++) {
            data[i] += data[i+stride];
        }
    }
    return recursiveReduce(data, stride);
}

__global__ void warmUp(int* idata, int* odata, int size) {
    int tid = threadIdx.x;
    int* data = idata + blockDim.x * blockIdx.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid%(stride*2)) == 0) {
            data[tid] += data[tid+stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        odata[blockIdx.x] = data[0];
    }
}

__global__ void reduceNeighboredLess(int* idata, int* odata, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) return;
    int* data = idata + blockDim.x * blockIdx.x;
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            data[index] += data[index+stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        odata[blockIdx.x] = data[0];
    }
}

__global__ void reduceInterleaved(int* idata, int* odata, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= size) return;
    int* data = idata + blockDim.x * blockIdx.x;
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            data[tid] += data[tid+stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        odata[blockIdx.x] = data[0];
    }
}

__global__ void reduceUnroll2(int* idata, int* odata, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 2;
    if (idx >= size) return;
    int* data = idata + blockDim.x * blockIdx.x * 2;
    if (idx + blockDim.x < size) {
        idata[idx] += idata[idx+blockDim.x];
    }
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            data[tid] += data[tid+stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        odata[blockIdx.x] = data[0];
    }
}

__global__ void reduceUnroll4(int* idata, int* odata, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 4;
    if (idx >= size) return;
    int* data = idata + blockDim.x * blockIdx.x * 4;
    if (idx + 3 * blockDim.x < size) {
        idata[idx] += idata[idx+blockDim.x];
        idata[idx] += idata[idx+blockDim.x*2];
        idata[idx] += idata[idx+blockDim.x*3];
    }
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            data[tid] += data[tid+stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        odata[blockIdx.x] = data[0];
    }
}

__global__ void reduceUnroll8(int* idata, int* odata, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 8;
    if (idx >= size) return;
    int* data = idata + blockIdx.x * blockDim.x * 8;
    if (idx + 7 * blockDim.x < size) {
        idata[idx] += idata[idx + blockDim.x];
        idata[idx] += idata[idx + blockDim.x * 2];
        idata[idx] += idata[idx + blockDim.x * 3];
        idata[idx] += idata[idx + blockDim.x * 4];
        idata[idx] += idata[idx + blockDim.x * 5];
        idata[idx] += idata[idx + blockDim.x * 6];
        idata[idx] += idata[idx + blockDim.x * 7];
    }
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            data[tid] += data[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        odata[blockIdx.x] = data[0];
    }
}

__global__ void reduceUnrollWarp8(int* idata, int* odata, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 8;
    if (idx >= size) return;
    int *data = idata + blockDim.x * blockIdx.x * 8;
    if (idx + 7 * blockDim.x < size) {
        idata[idx] += idata[idx+blockDim.x];
        idata[idx] += idata[idx+blockDim.x * 2];
        idata[idx] += idata[idx+blockDim.x * 3];
        idata[idx] += idata[idx+blockDim.x * 4];
        idata[idx] += idata[idx+blockDim.x * 5];
        idata[idx] += idata[idx+blockDim.x * 6];
        idata[idx] += idata[idx+blockDim.x * 7];
    }
    __syncthreads();
    for (int stride = blockDim.x/2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            data[tid] += data[tid+stride];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile int* vmem = data;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }
    if (tid == 0) {
        odata[blockIdx.x] = data[0];
    }
}

__global__ void reduceCompleteUnrollWarp8(int* idata, int* odata, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 8;
    if (idx >= size) return;
    int* data = idata + blockDim.x * blockIdx.x * 8;
    if (idx + 7 * blockDim.x < size) {
        idata[idx] += idata[idx + blockDim.x];
        idata[idx] += idata[idx + blockDim.x * 2];
        idata[idx] += idata[idx + blockDim.x * 3];
        idata[idx] += idata[idx + blockDim.x * 4];
        idata[idx] += idata[idx + blockDim.x * 5];
        idata[idx] += idata[idx + blockDim.x * 6];
        idata[idx] += idata[idx + blockDim.x * 7];
    }
    __syncthreads();
    if (blockDim.x >= 1024 && tid < 512) {
        data[tid] += data[tid+512];
    }
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) {
        data[tid] += data[tid+256];
    }
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) {
        data[tid] += data[tid+128];
    }
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) {
        data[tid] += data[tid+64];
    }
    __syncthreads();
    if (tid < 32) {
        volatile int* vmem = data;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }
    if (tid == 0) {
        odata[blockIdx.x] = data[0];
    }
}

__global__ void ReduceCompeleteUnrollWarp8Share(int* idata, int* odata, int size) {
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x * 8;
    __shared__ int smem[32];
    int* data = idata + blockDim.x * blockIdx.x * 8;
    int tmp_sum = 0;
    if (idx < size) {
        tmp_sum += data[tid];
        tmp_sum += data[tid + blockDim.x];
        tmp_sum += data[tid + blockDim.x * 2];
        tmp_sum += data[tid + blockDim.x * 3];
        tmp_sum += data[tid + blockDim.x * 4];
        tmp_sum += data[tid + blockDim.x * 5];
        tmp_sum += data[tid + blockDim.x * 6];
        tmp_sum += data[tid + blockDim.x * 7];
    }
    smem[tid] = tmp_sum;
    __syncthreads();
    if (blockDim.x >= 1024 && tid < 512) {
        smem[tid] += smem[tid+512];
    }
    __syncthreads();
    if (blockDim.x >= 512 && tid < 256) {
        smem[tid] += smem[tid+256];
    }
    __syncthreads();
    if (blockDim.x >= 256 && tid < 128) {
        smem[tid] += smem[tid+128];
    }
    __syncthreads();
    if (blockDim.x >= 128 && tid < 64) {
        smem[tid] += smem[tid+64];
    }
    __syncthreads();
    if (tid < 32) {
        volatile int* vmem = smem;
        vmem[tid] += vmem[tid+32];
        vmem[tid] += vmem[tid+16];
        vmem[tid] += vmem[tid+8];
        vmem[tid] += vmem[tid+4];
        vmem[tid] += vmem[tid+2];
        vmem[tid] += vmem[tid+1];
    }
    if (tid == 0) {
        odata[blockIdx.x] = smem[0];
    }
}

int main(int argc, char** argv) {
	initDevice(0);
	
	bool bResult = false;
	//initialization

	int size = 1 << 24;
	printf("	with array size %d  ", size);

	//execution configuration
	int blocksize = 1024;
	if (argc > 1)
	{
		blocksize = atoi(argv[1]);
	}
	dim3 block(blocksize, 1);
	dim3 grid((size - 1) / block.x + 1, 1);
	printf("grid %d block %d \n", grid.x, block.x);

	//allocate host memory
	size_t bytes = size * sizeof(int);
	int *idata_host = (int*)malloc(bytes);
	int *odata_host = (int*)malloc(grid.x * sizeof(int));
	int * tmp = (int*)malloc(bytes);

	//initialize the array
	initData_int(idata_host, size);

	memcpy(tmp, idata_host, bytes);
	double iStart, iElaps;
	int gpu_sum = 0;

	// device memory
	int * idata_dev = NULL;
	int * odata_dev = NULL;
	CHECK(cudaMalloc((void**)&idata_dev, bytes));
	CHECK(cudaMalloc((void**)&odata_dev, grid.x * sizeof(int)));

	//cpu reduction
	int cpu_sum = 0;
	iStart = cpuSecond();
	//cpu_sum = recursiveReduce(tmp, size);
	for (int i = 0; i < size; i++)
		cpu_sum += tmp[i];
	printf("cpu sum:%d \n", cpu_sum);
	iElaps = cpuSecond() - iStart;
	printf("cpu reduce                 elapsed %lf ms cpu_sum: %d\n", iElaps, cpu_sum);


	//kernel 1:reduceNeighbored

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	warmUp <<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu warmup                 elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceNeighboredLess<<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceNeighboredLess   elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceInterleaved<<<grid, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceInterleaved      elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceUnroll2<<<grid.x/2, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/2; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceUnroll2          elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x/2, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceUnroll4<<<grid.x/4, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/4; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceUnroll4          elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x/4, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceUnroll8<<<grid.x/8, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/8; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceUnroll8          elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x/8, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceUnrollWarp8<<<grid.x/8, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/8; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceUnrollWarp8      elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x/8, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	reduceCompleteUnrollWarp8<<<grid.x/8, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/8; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceCompleteUnrollWarp8      elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x/8, block.x);

	CHECK(cudaMemcpy(idata_dev, idata_host, bytes, cudaMemcpyHostToDevice));
	CHECK(cudaDeviceSynchronize());
	iStart = cpuSecond();
	ReduceCompeleteUnrollWarp8Share<<<grid.x/8, block >>>(idata_dev, odata_dev, size);
	cudaDeviceSynchronize();
	iElaps = cpuSecond() - iStart;
	cudaMemcpy(odata_host, odata_dev, grid.x * sizeof(int), cudaMemcpyDeviceToHost);
	gpu_sum = 0;
	for (int i = 0; i < grid.x/8; i++)
		gpu_sum += odata_host[i];
	printf("gpu reduceCompleteUnrollWarp8Share      elapsed %lf ms gpu_sum: %d <<<grid %d block %d>>>\n",
		iElaps, gpu_sum, grid.x/8, block.x);

    return 0;
}