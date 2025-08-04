#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

__global__ void simpleKernel(const float* A, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = sinf(A[i]);  // any device operation
    }
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    for (int i = 0; i < N; ++i) {
        h_A[i] = float(i);
    }

    float *d_A, *d_C;
    cudaError_t err;

    err = cudaMalloc((void**)&d_A, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_A failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMalloc((void**)&d_C, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMalloc d_C failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to device failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    simpleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_C, N);

    err = cudaGetLastError();  // catch kernel launch errors here
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy to host failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Optional: print a few results
    for (int i = 0; i < 5; ++i) {
        printf("C[%d] = %f\n", i, h_C[i]);
    }

    cudaFree(d_A);
    cudaFree(d_C);
    free(h_A);
    free(h_C);
    return 0;
}

