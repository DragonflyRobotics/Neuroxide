#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

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

extern "C" {
void matmul(const int M, const int N, const int K, float *A, float *B, float **C) {
    cublasHandle_t handle;
    cudaError_t cudaStat;
    cublasStatus_t stat;

    // Host matrices
    // float *h_A = (float *)malloc(M * K * sizeof(float)); // Matrix A
    // float *h_B = (float *)malloc(K * N * sizeof(float)); // Matrix B
    // float *h_C = (float *)malloc(M * N * sizeof(float)); // Matrix C

    // Verify that allocations succeeded
    // if (h_A == NULL || h_B == NULL || h_C == NULL)
    // {
    //     fprintf(stderr, "Failed to allocate host vectors!\n");
    //     exit(EXIT_FAILURE);
    // }

    // Initialize matrices A and B
    // memcpy(h_A, A, M * K * sizeof(float));
    // memcpy(h_B, B, K * N * sizeof(float));

    // Allocate device memory
    float *d_A = A;
    float *d_B = B;
    float *d_C;
    // cudaStat = cudaMalloc((void**)&d_A, M * K * sizeof(float));
    // cudaStat = cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaStat = cudaMalloc((void**)&d_C, M * N * sizeof(float));


    // Copy matrices A and B to device (cuBLAS expects column-major format)
    // stat = cublasSetMatrix(M, K, sizeof(float), h_A, M, d_A, M);  // Copy A
    // stat = cublasSetMatrix(K, N, sizeof(float), h_B, K, d_B, K);  // Copy B

    // Initialize cuBLAS handle
    stat = cublasCreate(&handle);

    // Perform matrix multiplication: C = alpha * A * B + beta * C
    float alpha = 1.0f, beta = 0.0f;
    stat = cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, N, K,  // Use transposed (T) for row-major
                       &alpha,        // Alpha
                       d_A, K,        // A and leading dimension K (for row-major)
                       d_B, N,        // B and leading dimension N (for row-major)
                       &beta,         // Beta
                       d_C, M);       // C and leading dimension M
    // Transpose d_C from column-major to row-major using cublasSgeam
    // Allocate memory for the transposed matrix
    float *d_C_transposed;
    cudaMalloc((void**)&d_C_transposed, M * N * sizeof(float));

    dim3 blockDim(16, 16);
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (M + blockDim.y - 1) / blockDim.y);

    // Launch transpose kernel
    transpose<<<gridDim, blockDim>>>(d_C, d_C_transposed, M, N);

    if (stat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "cuBLAS SGEAM failed: " << stat << std::endl;
    }
    if (cudaStat != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(cudaStat) << std::endl;
    }
    // transpose d_C
    // Copy result matrix C back to host
    // stat = cublasGetMatrix(M, N, sizeof(float), d_C_transposed, M, h_C, M);
    *C = d_C_transposed;

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    // Free device memory
    // cudaFree(d_A);
    // cudaFree(d_B);
    // cudaFree(d_C);
    //
    // memcpy(C, h_C, M * N * sizeof(float));
}
}
