#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

bool verifyAllocations(float* var)
{
    if (var == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }
    return true;
}

bool checkCUDASuccess(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    return true;
}

float* allocateCUDAMemory(float* var, size_t size)
{
    float *d_var = NULL;
    checkCUDASuccess(cudaMalloc((void **)&d_var, size));
    return d_var;
}

void copyDataToCUDAMemory(float* d_var, const float* h_var, size_t size)
{
    checkCUDASuccess(cudaMemcpy(d_var, h_var, size, cudaMemcpyHostToDevice));
}

void copyDataToHostMemory(float* h_var, float* d_var, size_t size)
{
    checkCUDASuccess(cudaMemcpy(h_var, d_var, size, cudaMemcpyDeviceToHost));
}

