#ifndef HANDLES_CUH
#define HANDLES_CUH

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cutensor.h>

#include "pool.cuh"

struct Handles {
    cublasHandle_t cublas_handle;
    cutensorHandle_t cutensor_handle;
    cudaDeviceProp prop;

    Handles(int device = 0) {
        cudaGetDeviceProperties(&prop, device);
        cublasCreate(&cublas_handle);
        cublasSetStream(cublas_handle, pool.get_stream());
        cutensorCreate(&cutensor_handle);
    }

    ~Handles() {
        cutensorDestroy(cutensor_handle);
        cublasDestroy(cublas_handle);
    }
};

inline Handles handles = Handles();  

#endif
