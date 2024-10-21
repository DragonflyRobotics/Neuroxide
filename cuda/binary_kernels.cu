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
vectorAdd(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] + B[i];
    }
}

__global__ void
vectorSub(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] - B[i];
    }
}

__global__ void
vectorMul(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] * B[i];
    }
}

__global__ void
vectorDiv(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = A[i] / B[i];
    }
}

__global__ void
vectorPow(const float *A, const float *B, float *C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < numElements)
    {
        C[i] = pow(A[i], B[i]);
    }
}

/**
 * Host main routine
 */
extern  "C" {
    void add_kernel(const int len, const float* A, const float* B, float* C)
    {
        binaryVectorOp(len, A, B, C, vectorAdd);
    }
    void sub_kernel(const int len, const float* A, const float* B, float* C)
    {
        binaryVectorOp(len, A, B, C, vectorSub);
    }
    void mul_kernel(const int len, const float* A, const float* B, float* C)
    {
        binaryVectorOp(len, A, B, C, vectorMul);
    }
    void div_kernel(const int len, const float* A, const float* B, float* C)
    {
        binaryVectorOp(len, A, B, C, vectorDiv);
    }
    void pow_kernel(const int len, const float* A, const float* B, float* C)
    {
        binaryVectorOp(len, A, B, C, vectorPow);
    }
}


