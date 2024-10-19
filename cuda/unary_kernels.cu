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

// For the CUDA runtime routines (prefixed with "cuda_")
#include <cuda_runtime.h>

#include "utils.cuh"


/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 */
__global__ void
vectorSin(const float *A, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = sin(A[i]);
    }
}

__global__ void
vectorCos(const float *A, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = cos(A[i]);
    }
}

__global__ void
vectorLn(const float *A, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = log(A[i]);
    }
}


/**
 * Host main routine
 */
extern  "C" {
    void sin_kernel(const int len, const float* A, float* C)
    {
        unaryVectorOp(len, A, C, vectorSin);
    }
    void cos_kernel(const int len, const float* A, float* C)
    {
        unaryVectorOp(len, A, C, vectorCos);
    }
    void ln_kernel(const int len, const float* A, float* C)
    {
        unaryVectorOp(len, A, C, vectorLn);
    }
}


