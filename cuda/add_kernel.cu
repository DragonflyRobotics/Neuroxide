/**
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/**
 * Vector addition: C = A + B.
 *
 * This sample is a very basic sample that implements element by element
 * vector addition. It is the same as the sample illustrating Chapter 2
 * of the programming guide with some additions like error checking.
 */

#include <stdio.h>

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "utils.h"


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * Host main routine
 */
extern  "C" {

int vectorAdd_main(const int len, const float* A, const float* B, const float* C)
{
    long size = len * sizeof(float);
    // Allocate the host input vector A
    float *h_A = (float *)malloc(size);

    // Allocate the host input vector B
    float *h_B = (float *)malloc(size);

    // Allocate the host output vector C
    float *h_C = (float *)malloc(size);

    // Verify that allocations succeeded
    if (h_A == NULL || h_B == NULL || h_C == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    memcpy(h_A, A, size);
    memcpy(h_B, B, size);

    float *d_A = allocateCUDAMemory(h_A, size);
    float *d_B = allocateCUDAMemory(h_B, size);
    float *d_C = allocateCUDAMemory(h_C, size);
    

    copyDataToCUDAMemory(d_A, h_A, size);
    copyDataToCUDAMemory(d_B, h_B, size);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =(len + threadsPerBlock - 1) / threadsPerBlock;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, len);
    checkCUDASuccess(cudaGetLastError());

    // Copy the device result vector in device memory to the host result vector
    copyDataToHostMemory(h_C, d_C, size);
    for (int i = 0; i < len; ++i)
    {
        printf("C[%d] = %f\n", i, h_C[i]);
    }
    

    checkCUDASuccess(cudaFree(d_A));
    checkCUDASuccess(cudaFree(d_B));
    checkCUDASuccess(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
}


