#define CHECK(call) \
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess) {\
        printf("ERROR: %s, %d\n", __FILE__, __LINE__);\
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>

#include <cuda_runtime.h>
double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec+(double)tp.tv_usec*1e-6);
}

void initData(float* idata, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        idata[i] = (float)(rand()&0xffff)/1000.0f;
    }
}

void initData_inf(int* idata, int size) {
    time_t t;
    srand((unsigned)time(&t));
    for (int i = 0; i < size; i++) {
        idata[i] = (int)(rand()&0xff);
    }
}

void printMatrix(float* C, const int nx, const int ny) {
    float* ic = C;
    printf("Matrix<%d, %d>: ", nx, ny);
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            printf("%6f ", ic[j]);
        }
        ic+=nx;
        printf("\n");
    }
}

void initDevice(int devNum) {
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, devNum));
    printf("Using device %d: %s\n", devNum, deviceProp.name);
    CHECK(cudaSetDevice(devNum));
}

void checkResult(float* hostRef, float* devRef, const int N) {
    double epsilon = 1.0e-8;
    for (int i = 0; i < N; i++) {
        if (abs(hostRef[i] - devRef[i]) > epsilon) {
            printf("res not match\n");
            printf("hostRef[%d]: %f, devRef[%d]: %f\n", i, hostRef[i], i, devRef[i]);
            return;
        }
    }
    printf("check result success!\n");
}