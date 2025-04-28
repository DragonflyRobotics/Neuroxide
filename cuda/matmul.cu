#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include "pool.cuh"

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))  // Macro for column-major indexing

__global__ void transpose(float *input, float *output, int rows, int cols) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < cols && y < rows) {
        int input_idx = y + x * rows;     // Column-major index
        int output_idx = x + y * cols;   // Row-major index
        output[output_idx] = input[input_idx];
    }
}

static cublasHandle_t handle = 0; 
cublasStatus_t stat = cublasCreate(&handle);

extern "C" {
void matmul(const int M, const int N, const int K, float *A, float *B, float **C) {
    // cublasHandle_t handle;
    cudaError_t cudaStat;
    cublasStatus_t stat;

    // Allocate device memory
    float *d_A = A;
    float *d_B = B;
    float *d_C;
    // cudaStat = cudaMalloc((void**)&d_C, M * N * sizeof(float));

    //cudamallocasync 
    //use pool 
    d_C = pool.malloc(M * N * sizeof(float));

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    float alpha = 1.0f, beta = 0.0f;
    stat = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,  // Use transposed (T) for row-major
                       &alpha,        // Alpha
                       d_B, N,        // A and leading dimension K (for row-major)
                       d_A, K,        // B and leading dimension N (for row-major)
                       &beta,         // Beta
                       d_C, N);       // C and leading dimension M
    *C = d_C;


    // Free device memory
    // cudaFree(d_A);
    // cudaFree(d_B);
}
}
