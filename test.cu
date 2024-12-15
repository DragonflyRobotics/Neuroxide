#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include "cublas_v2.h"

#define IDX2C(i, j, ld) (((j) * (ld)) + (i))  // Macro for column-major indexing
#define M 2  // Rows of A and C
#define N 3  // Columns of B and C
#define K 2  // Columns of A and rows of B

int main() {
    cublasHandle_t handle;
    cudaError_t cudaStat;
    cublasStatus_t stat;

    // Initialize matrices in host memory (row-major order)
    float h_A[M * K] = {1.0f, 4.0f,  // Matrix A (2x2) - Row-major
                        2.0f, 5.0f};
    float h_B[K * N] = {7.0f, 9.0f, 11.0f,  // Matrix B (2x3) - Row-major
                        8.0f, 10.0f, 12.0f};

    float h_C[M * N];  // Result matrix C (2x3)

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaStat = cudaMalloc((void**)&d_A, M * K * sizeof(float));
    cudaStat = cudaMalloc((void**)&d_B, K * N * sizeof(float));
    cudaStat = cudaMalloc((void**)&d_C, M * N * sizeof(float));

    // Copy matrices A and B to device (cuBLAS expects column-major format)
    stat = cublasSetMatrix(M, K, sizeof(float), h_A, M, d_A, M);  // Copy A
    stat = cublasSetMatrix(K, N, sizeof(float), h_B, K, d_B, K);  // Copy B

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

    // Copy result matrix C back to host
    stat = cublasGetMatrix(M, N, sizeof(float), d_C, M, h_C, M);

    // Destroy cuBLAS handle
    cublasDestroy(handle);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Print result matrix C
    printf("Matrix C (result of A*B):\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%8.2f", h_C[IDX2C(i, j, M)]);
        }
        printf("\n");
    }

    return 0;
}

