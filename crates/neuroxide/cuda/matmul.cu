#include <cstdio>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <vector>
#include "pool.cuh"
#include "utils.cuh"
#include "handles.cuh"

using namespace std;
using std::chrono::high_resolution_clock;
using std::chrono::duration_cast;
using std::chrono::duration;
using std::chrono::milliseconds;

// #define IDX2C(i, j, ld) (((j) * (ld)) + (i))  // Macro for column-major indexing
//
// __global__ void transpose(float *input, float *output, int rows, int cols) {
//     int x = blockIdx.x * blockDim.x + threadIdx.x;
//     int y = blockIdx.y * blockDim.y + threadIdx.y;
//
//     if (x < cols && y < rows) {
//         int input_idx = y + x * rows;     // Column-major index
//         int output_idx = x + y * cols;   // Row-major index
//         output[output_idx] = input[input_idx];
//     }
// }


extern "C" {
void matmul(const int M, const int N, const int K, float *A, float *B, float **C) {
    cublasHandle_t handle = handles.cublas_handle;
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

void b_matmul(const int D, const int M, const int N, const int K, const int a_broad, const int b_broad, float *A, float *B, float **C) {
    cublasHandle_t handle = handles.cublas_handle;
    // Shape of A (D, M, K), B (D, K, N), C (D, M, N)
    cudaError_t cudaStat;
    cublasStatus_t stat;

    // Allocate device memory
    float *d_A = A;
    float *d_B = B;
    float *d_C;
    // cudaStat = cudaMalloc((void**)&d_C, B * M * N * sizeof(float));
    // float *h_A = (float*) malloc(6 * sizeof(float));
    // cudaStat = cudaMemcpy(h_A, d_A, 6 * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i=0; i<6; i++) {
    //     std::cout << "h_A[" << i << "] = " << h_A[i] << std::endl;
    // }
    // float *h_B = (float*) malloc(36 * sizeof(float));
    // cudaStat = cudaMemcpy(h_B, d_B, 36 * sizeof(float), cudaMemcpyDeviceToHost);
    // for (int i=0; i<36; i++) {
    //     std::cout << "h_B[" << i << "] = " << h_B[i] << std::endl;
    // }

    //cudamallocasync 
    //use pool 
    d_C = pool.malloc(D * M * N * sizeof(float));


    std::vector<const float*> A_array(D);
    std::vector<const float*> B_array(D);
    std::vector<float*> C_array(D);

    // for (int i = 0; i < D; i++) {
    //     A_array[i] = d_A + i * M * K;
    //     B_array[i] = d_B + i * K * N;
    //     C_array[i] = d_C + i * M * N;
    // }
    // Calculate pointer strides
    for (int i = 0; i < D; i++) {
        A_array[i] = d_A + ((a_broad == 1) ? i * M*K : M * K);
        B_array[i] = d_B + ((b_broad == 1) ? i * K*N : K * N);
        C_array[i] = d_C + i * M * N;
    }

    // float **d_A_array, **d_B_array, **d_C_array;
    // checkCUDASuccess(cudaMalloc(&d_A_array, D * sizeof(float*)));
    // checkCUDASuccess(cudaMalloc(&d_B_array, D * sizeof(float*)));
    // checkCUDASuccess(cudaMalloc(&d_C_array, D * sizeof(float*)));
    float **d_A_array, **d_B_array, **d_C_array;

    d_A_array = (float**) pool.malloc(D * sizeof(float*));
    d_B_array = (float**) pool.malloc(D * sizeof(float*));
    d_C_array = (float**) pool.malloc(D * sizeof(float*));

    checkCUDASuccess(cudaMemcpy(d_A_array, A_array.data(), D * sizeof(float*), cudaMemcpyHostToDevice));
    checkCUDASuccess(cudaMemcpy(d_B_array, B_array.data(), D * sizeof(float*), cudaMemcpyHostToDevice));
    checkCUDASuccess(cudaMemcpy(d_C_array, C_array.data(), D * sizeof(float*), cudaMemcpyHostToDevice));


    // print d_A_array, d_B_array, d_C_array
    // std::vector<const float*> h_A_array(D);
    // checkCUDASuccess(cudaMemcpy(h_A_array.data(), d_A_array, D * sizeof(float*), cudaMemcpyDeviceToHost));

    // std::cout << "=== Printing A batch matrices ===\n";
    // for (int d = 0; d < D; ++d) {
    //     std::vector<float> h_A_matrix(M * K);  // single matrix A_i
    //     checkCUDASuccess(cudaMemcpy(h_A_matrix.data(), h_A_array[d], M * K * sizeof(float), cudaMemcpyDeviceToHost));
    //
    //     std::cout << "Batch " << d << " A matrix:\n";
    //     for (int i = 0; i < M; ++i) {
    //         for (int j = 0; j < K; ++j) {
    //             std::cout << h_A_matrix[i * K + j] << " ";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << std::endl;
    // }
    // std::cout << "=== Printing B batch matrices ===\n";
    // std::vector<const float*> h_B_array(D);
    // checkCUDASuccess(cudaMemcpy(h_B_array.data(), d_B_array, D * sizeof(float*), cudaMemcpyDeviceToHost));
    // for (int d = 0; d < D; ++d) {
    //     std::vector<float> h_B_matrix(K * N);  // single matrix B_i
    //     checkCUDASuccess(cudaMemcpy(h_B_matrix.data(), h_B_array[d], K * N * sizeof(float), cudaMemcpyDeviceToHost));
    //
    //     std::cout << "Batch " << d << " B matrix:\n";
    //     for (int i = 0; i < K; ++i) {
    //         for (int j = 0; j < N; ++j) {
    //             std::cout << h_B_matrix[i * N + j] << " ";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << std::endl;
    // }


    // Perform matrix multiplication: C = alpha * A * B + beta * C
    float alpha = 1.0f, beta = 0.0f;
    stat = cublasSgemmBatched(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                              &alpha, d_B_array, N, d_A_array, K,
                              &beta, d_C_array, N, D);
    *C = d_C;

    
    // std::vector<float> h_C(D * M * N, 0.0f);
    // checkCUDASuccess(cudaMemcpy(h_C.data(), d_C, h_C.size() * sizeof(float), cudaMemcpyDeviceToHost));
    //
    // // Print results
    // for (int d = 0; d < D; ++d) {
    //     std::cout << "Batch " << d << " result:\n";
    //     for (int i = 0; i < M; ++i) {
    //         for (int j = 0; j < N; ++j) {
    //             std::cout << h_C[d * M * N + i * N + j] << " ";
    //         }
    //         std::cout << "\n";
    //     }
    //     std::cout << std::endl;
    // }
    // Free device memory
    // cudaFree(d_A);
    // cudaFree(d_B);
}
}
