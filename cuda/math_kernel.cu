#include <cuda_runtime.h>
#include <math.h>

extern "C" __global__ void math_operations(float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = sinf(input[idx]) + cosf(input[idx]) + sqrtf(input[idx]);
    }
}

extern "C" void launch_math_operations(float* input, float* output, int N) {
    float* d_input;
    float* d_output;

    cudaMalloc((void**)&d_input, N * sizeof(float));
    cudaMalloc((void**)&d_output, N * sizeof(float));

    cudaMemcpy(d_input, input, N * sizeof(float), cudaMemcpyHostToDevice);

    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;

    math_operations<<<blocks_per_grid, threads_per_block>>>(d_input, d_output, N);

    cudaMemcpy(output, d_output, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
}

