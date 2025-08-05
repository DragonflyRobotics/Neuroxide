#include <cstdio>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <iostream>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

#include "utils.cuh"

#define HANDLE_ERROR(x)                                                   \
{ auto const __err = x;                                                   \
    if( __err != CUTENSOR_STATUS_SUCCESS )                                  \
    { printf("Error: %d %s\n", __LINE__, cutensorGetErrorString(__err)); exit(-1); } \
};

typedef float floatTypeA;
typedef float floatTypeC;
typedef float floatTypeCompute;

cutensorDataType_t          const typeA       = CUTENSOR_R_32F;
cutensorDataType_t          const typeC       = CUTENSOR_R_32F;
cutensorComputeDescriptor_t const descCompute = CUTENSOR_COMPUTE_DESC_32F;
floatTypeCompute alpha = (floatTypeCompute)1.0f;

extern "C" {
void transpose(const int M, const int N, float *A, float **C) {
    std::vector<int> modeA{'w','h'};
    std::vector<int> modeC{'h','w'};
    int nmodeA = modeA.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    extent['w'] = M;
    extent['h'] = N;

    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);

    /**********************
     * Allocating data
     **********************/

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    void *A_d = A;
    void *C_d;
    checkCUDASuccess(cudaMalloc((void**) &C_d, sizeC));

    uint32_t const kAlignment = 256;  // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    std::vector<int64_t> strideA = {extentA[1], 1};  // B, H, W
    std::vector<int64_t> strideC = {extentC[1], 1};  // B, H, W

    /*************************
     * CUTENSOR
     *************************/

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t  descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descA,
                                                nmodeA,
                                                extentA.data(),
                                                strideA.data(),
                                                typeA,
                                                kAlignment));

    cutensorTensorDescriptor_t  descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descC,
                                                nmodeC,
                                                extentC.data(),
                                                strideC.data(),
                                                typeC,
                                                kAlignment));

    /*******************************
     * Create Permutation Descriptor
     *******************************/

    cutensorOperationDescriptor_t  desc;
    HANDLE_ERROR(cutensorCreatePermutation(handle,
                                           &desc,
                                           descA,
                                           modeA.data(),
                                           CUTENSOR_OP_IDENTITY,
                                           descC,
                                           modeC.data(),
                                           descCompute));


    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                         (void*)&scalarType,
                                                         sizeof(scalarType)));

    assert(scalarType == CUTENSOR_R_32F);


    /**************************
    * Set the algorithm to use
    ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t  planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle,
                                              &planPref,
                                              algo,
                                              CUTENSOR_JIT_MODE_NONE));

    /**************************
     * Create Plan
     **************************/

    cutensorPlan_t  plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                                    &plan,
                                    desc,
                                    planPref,
                                    0 /* workspaceSizeLimit */));

    /**********************
     * Run
     **********************/


    HANDLE_ERROR(cutensorPermute(handle,
                                 plan,
                                 &alpha, A_d, C_d, nullptr /* stream */));

    floatTypeC *temp;
    checkCUDASuccess(cudaMallocHost((void**) &temp, sizeof(floatTypeC) * elementsC));
    checkCUDASuccess(cudaMemcpy(temp, C_d, sizeC, cudaMemcpyDeviceToHost));
    for (int i = 0; i < elementsC; i++) {
        std::cout << "C[" << i << "] = " << temp[i] << std::endl;
    }
    *C = (float*)C_d;

    /*************************/


    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyPlanPreference(planPref));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
}

void transpose_b(const int D, const int M, const int N, float *A, float **C) {
    std::vector<int> modeA{'d', 'w','h'};
    std::vector<int> modeC{'d', 'h','w'};
    int nmodeA = modeA.size();
    int nmodeC = modeC.size();

    std::unordered_map<int, int64_t> extent;
    extent['d'] = D;
    extent['w'] = M;
    extent['h'] = N;

    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);
    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);

    /**********************
     * Allocating data
     **********************/

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    void *A_d = A;
    void *C_d;
    checkCUDASuccess(cudaMalloc((void**) &C_d, sizeC));

    uint32_t const kAlignment = 256;  // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    std::vector<int64_t> strideA = {extentA[1] * extentA[2], extentA[2], 1};  // D, H, W
    std::vector<int64_t> strideC = {extentC[1] * extentC[2], extentC[2], 1};  // D, H, W

    /*************************
     * CUTENSOR
     *************************/

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t  descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descA,
                                                nmodeA,
                                                extentA.data(),
                                                strideA.data(),
                                                typeA,
                                                kAlignment));

    cutensorTensorDescriptor_t  descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                                                &descC,
                                                nmodeC,
                                                extentC.data(),
                                                strideC.data(),
                                                typeC,
                                                kAlignment));

    /*******************************
     * Create Permutation Descriptor
     *******************************/

    cutensorOperationDescriptor_t  desc;
    HANDLE_ERROR(cutensorCreatePermutation(handle,
                                           &desc,
                                           descA,
                                           modeA.data(),
                                           CUTENSOR_OP_IDENTITY,
                                           descC,
                                           modeC.data(),
                                           descCompute));


    /*****************************
     * Optional (but recommended): ensure that the scalar type is correct.
     *****************************/

    cutensorDataType_t scalarType;
    HANDLE_ERROR(cutensorOperationDescriptorGetAttribute(handle, desc,
                                                         CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
                                                         (void*)&scalarType,
                                                         sizeof(scalarType)));

    assert(scalarType == CUTENSOR_R_32F);


    /**************************
    * Set the algorithm to use
    ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t  planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(handle,
                                              &planPref,
                                              algo,
                                              CUTENSOR_JIT_MODE_NONE));

    /**************************
     * Create Plan
     **************************/

    cutensorPlan_t  plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                                    &plan,
                                    desc,
                                    planPref,
                                    0 /* workspaceSizeLimit */));

    /**********************
     * Run
     **********************/


    HANDLE_ERROR(cutensorPermute(handle,
                                 plan,
                                 &alpha, A_d, C_d, nullptr /* stream */));

    floatTypeC *temp;
    checkCUDASuccess(cudaMallocHost((void**) &temp, sizeof(floatTypeC) * elementsC));
    checkCUDASuccess(cudaMemcpy(temp, C_d, sizeC, cudaMemcpyDeviceToHost));
    for (int i = 0; i < elementsC; i++) {
        std::cout << "C[" << i << "] = " << temp[i] << std::endl;
    }
    *C = (float*)C_d;

    /*************************/


    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyPlanPreference(planPref));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
}
}
