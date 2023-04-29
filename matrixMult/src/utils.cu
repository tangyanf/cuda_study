#include "utils.cuh"
#include "kernel.cuh"

void randomize_matrix(float* mat, int N) {
    struct timeval time;
    gettimeofday(&time, NULL);
    srand(time.tv_usec);
    for (int i = 0; i < N; i++) {
        float tmp = (float)(rand() % 5) + 0.01 * (rand()%5);
        tmp = (rand()%2 == 0) ? tmp : tmp * (-1.);
        mat[i] = tmp;
    }
}

void copy_matrix(float* src, float* dest, int N) {
    int i;
    for (i = 0; i < N; i++) {
        *(dest + i) = *(src + i);
    }
    if (i != N) {
        printf("copy failed at %d while there are %d elements in total.\n", i, N);
    }
}

bool verify_matrix(float *mat1, float *mat2, int N) {
    double diff = 0.0;
    int i;
    for (i = 0; mat1 + i && mat2 + i && i < N; i++) {
        diff = fabs((double) mat1[i] - (double) mat2[i]);
        if (diff > 1e-2) {
            printf("error. %5.2f,%5.2f,%d\n", mat1[i], mat2[i], i);
            return false;
        }
    }
    return true;
}

void test_cublas(cublasHandle_t handle, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    //cublas列主序计算：https://www.cnblogs.com/cuancuancuanhao/p/7763256.html
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, B, N, A, K, &beta, C, N);
}

void test_mysgemm_v1(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    dim3 block(32,32);
    dim3 grid((N-1)/block.x+1, (M-1)/block.y+1);
    mysgemm_v1<<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v2(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    dim3 block(1024);
    dim3 grid((N-1)/32+1, (M-1)/32+1);
    mysgemm_v2<32><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

void test_mysgemm_v3(int M, int N, int K, float alpha, float* A, float* B, float beta, float* C) {
    dim3 block(512);
    dim3 grid((N-1)/64+1, (M-1)/64+1);
    mysgemm_v31<64,64,8,8><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
}

void test_kernel(int kernel_num, int M, int N, int K, float alpha, float *A, float *B, float beta, float *C,
                 cublasHandle_t handle) {
    switch (kernel_num) {
        case 0:
            test_cublas(handle, M, N, K, alpha, A, B, beta, C);
            break;
        case 1:
            test_mysgemm_v1(M, N, K, alpha, A, B, beta, C);
            break;
        case 2:
            test_mysgemm_v2(M, N, K, alpha, A, B, beta, C);
        case 3:
            test_mysgemm_v3(M, N, K, alpha, A, B, beta, C);
        default:
            break;
    }
}