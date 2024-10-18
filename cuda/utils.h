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

float* allocateCUDAMemory(size_t size)
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


int binaryVectorOp(const int len, const float* A, const float* B, float* C, void (*kernel)(const float*, const float*, float*, int))
{
    long size = len * sizeof(float);
    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    // float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    memcpy(h_A, A, size);
    memcpy(h_B, B, size);

    float *d_A = allocateCUDAMemory(size);
    float *d_B = allocateCUDAMemory(size);
    float *d_C = allocateCUDAMemory(size);


    copyDataToCUDAMemory(d_A, h_A, size);
    copyDataToCUDAMemory(d_B, h_B, size);

    // Launch the Vector Add CUDA Kernel
    // int threadsPerBlock = 256;
    
    // Query device properties
    cudaDeviceProp prop;
    checkCUDASuccess(cudaGetDeviceProperties(&prop, 0));

    // Set threadsPerBlock to a multiple of warp size (32) and within the max limit
    int threadsPerBlock = prop.maxThreadsPerBlock;
    if (threadsPerBlock > 1024) {
        threadsPerBlock = 1024;
    } else if (threadsPerBlock % 32 != 0) {
        threadsPerBlock = (threadsPerBlock / 32) * 32;
    }


    int blocksPerGrid =(len + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, len);
    checkCUDASuccess(cudaGetLastError());

    // Copy the device result vector in device memory to the host result vector
    copyDataToHostMemory(C, d_C, size);
    // for (int i = 0; i < len; ++i)
    // {
    //     printf("C[%d] = %f\n", i, C[i]);
    // }


    checkCUDASuccess(cudaFree(d_A));
    checkCUDASuccess(cudaFree(d_B));
    checkCUDASuccess(cudaFree(d_C));
    free(h_A);
    free(h_B);

    return 0;
}
