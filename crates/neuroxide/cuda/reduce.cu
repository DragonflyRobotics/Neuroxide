#include <iostream>
#include <ostream>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <cutensor.h>

#define HANDLE_ERROR(x)                                               \
{ const auto err = x;                                                 \
  if( err != CUTENSOR_STATUS_SUCCESS )                                \
  { printf("Error: %s\n", cutensorGetErrorString(err)); exit(-1); } \
};

#define HANDLE_CUDA_ERROR(x)                                      \
{ const auto err = x;                                             \
  if( err != cudaSuccess )                                        \
  { printf("Error: %s\n", cudaGetErrorString(err)); exit(-1); } \
};

typedef float floatTypeA;
typedef float floatTypeB;
typedef float floatTypeC;
typedef float floatTypeCompute;

cutensorDataType_t typeA = CUTENSOR_R_32F;
cutensorDataType_t typeC = CUTENSOR_R_32F;
const cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;

std::vector<int32_t> modeA{'b', 'm', 'n'};
std::vector<int32_t> modeC{'m','n'};
int32_t nmodeA = modeA.size();
int32_t nmodeC = modeC.size();

extern "C" {
void reduce(const int D, const int M, const int N, float *A, float **C)
{

    /**********************
     * Computing (partial) reduction : C_{m,v} = alpha * A_{m,h,k,v} + beta * C_{m,v}
     *********************/
    floatTypeCompute alpha = (floatTypeCompute)1.0f;
    floatTypeCompute beta  = (floatTypeCompute)0.f;


    std::unordered_map<int32_t, int64_t> extent;
    extent['b'] = D;
    extent['m'] = M;
    extent['n'] = N;

    std::vector<int64_t> extentC;
    for (auto mode : modeC)
        extentC.push_back(extent[mode]);
    std::vector<int64_t> extentA;
    for (auto mode : modeA)
        extentA.push_back(extent[mode]);

    std::vector<int64_t> strideA = {extentA[1]*extentA[2], extentA[2], 1};  // B, H, W
    std::vector<int64_t> strideC = {extentC[1], 1};  // B, H, W
    /**********************
     * Allocating data
     *********************/

    size_t elementsA = 1;
    for (auto mode : modeA)
        elementsA *= extent[mode];
    size_t elementsC = 1;
    for (auto mode : modeC)
        elementsC *= extent[mode];

    size_t sizeA = sizeof(floatTypeA) * elementsA;
    size_t sizeC = sizeof(floatTypeC) * elementsC;

    void *A_d = A;
    void *C_d = nullptr;
    HANDLE_CUDA_ERROR(cudaMalloc((void**)&C_d, sizeC));

    const uint32_t kAlignment = 256; // Alignment of the global-memory device pointers (bytes)
    // assert(uintptr_t(A_d) % kAlignment == 0);
    // assert(uintptr_t(C_d) % kAlignment == 0);

    floatTypeC *C_h = (floatTypeC*) malloc(sizeof(floatTypeC) * elementsC);


    /*******************
     * Initialize data
     *******************/


    for (int64_t i = 0; i < elementsC; i++) {
        C_h[i] = 0.0;
        std::cout << "C[" << i << "] = " << C_h[i] << std::endl;
    }

    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C_h, sizeC, cudaMemcpyHostToDevice));

    /*************************
     * cuTENSOR
     *************************/

    cutensorHandle_t handle;
    HANDLE_ERROR(cutensorCreate(&handle));

    /**********************
     * Create Tensor Descriptors
     **********************/

    cutensorTensorDescriptor_t descA;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                 &descA,
                 nmodeA,
                 extentA.data(),
                strideA.data(),
                 typeA, kAlignment));

    cutensorTensorDescriptor_t descC;
    HANDLE_ERROR(cutensorCreateTensorDescriptor(handle,
                 &descC,
                 nmodeC,
                 extentC.data(),
                strideC.data(),
                 typeC, kAlignment));

    const cutensorOperator_t opReduce = CUTENSOR_OP_ADD;

    /*******************************
     * Create Reduction Descriptor
     *******************************/

    cutensorOperationDescriptor_t desc;
    HANDLE_ERROR(cutensorCreateReduction(
                 handle, &desc,
                 descA, modeA.data(), CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(), CUTENSOR_OP_IDENTITY,
                 descC, modeC.data(),
                 opReduce, descCompute));

    /**************************
    * Set the algorithm to use
    ***************************/

    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;

    cutensorPlanPreference_t planPref;
    HANDLE_ERROR(cutensorCreatePlanPreference(
                               handle,
                               &planPref,
                               algo,
                               CUTENSOR_JIT_MODE_NONE));

    /**********************
     * Query workspace estimate
     **********************/

    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref = CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_ERROR(cutensorEstimateWorkspaceSize(handle,
                                          desc,
                                          planPref,
                                          workspacePref,
                                          &workspaceSizeEstimate));

    /**************************
     * Create Contraction Plan
     **************************/

    cutensorPlan_t plan;
    HANDLE_ERROR(cutensorCreatePlan(handle,
                 &plan,
                 desc,
                 planPref,
                 workspaceSizeEstimate));

    /**************************
     * Optional: Query information about the created plan
     **************************/

    // query actually used workspace
    uint64_t actualWorkspaceSize = 0;
    HANDLE_ERROR(cutensorPlanGetAttribute(handle,
        plan,
        CUTENSOR_PLAN_REQUIRED_WORKSPACE,
        &actualWorkspaceSize,
        sizeof(actualWorkspaceSize)));

    // At this point the user knows exactly how much memory is need by the operation and
    // only the smaller actual workspace needs to be allocated
    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    void *work = nullptr;
    if (actualWorkspaceSize > 0)
    {
        HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 == 0); // workspace must be aligned to 128 byte-boundary
    }

    /**********************
     * Run
     **********************/

    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));

    HANDLE_CUDA_ERROR(cudaMemcpy(C_d, C_h, sizeC, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());


    HANDLE_ERROR(cutensorReduce(handle, plan,
                                (const void*)&alpha, A_d,
                                (const void*)&beta,  C_d,
                                C_d, work, actualWorkspaceSize, stream));


    /*************************/

    // copy data back and print
    HANDLE_CUDA_ERROR(cudaMemcpy(C_h, C_d, sizeC, cudaMemcpyDeviceToHost));
    HANDLE_CUDA_ERROR(cudaDeviceSynchronize());
    printf("=== Result C matrix ===\n");
    for (size_t i = 0; i < elementsC; i++)
    {
        std::cout << "C[" << i << "] = " << C_h[i] << std::endl;
    }
    *C = C_h;


    HANDLE_ERROR(cutensorDestroy(handle));
    HANDLE_ERROR(cutensorDestroyPlan(plan));
    HANDLE_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_ERROR(cutensorDestroyTensorDescriptor(descC));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));

    // if (A) free(A);
    // if (C) free(C);
    // if (A_d) cudaFree(A_d);
    // if (C_d) cudaFree(C_d);
    // if (work) cudaFree(work);

}
}
