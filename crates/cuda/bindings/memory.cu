#include <cuda_runtime.h>
#include <stdio.h>

extern "C" {
// Allocate device memory
void *allocateDeviceMemory(size_t size) {
    void *d_ptr;
    cudaError_t err = cudaMalloc(&d_ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error allocating device memory: %s\n",
                cudaGetErrorString(err));
        return nullptr;
    }
    return d_ptr;
}

// Free device memory
void freeDeviceMemory(void *d_ptr) {
    cudaError_t err = cudaFree(d_ptr);
    if (err != cudaSuccess) {
        fprintf(stderr, "Error freeing device memory: %s\n",
                cudaGetErrorString(err));
    }
}
}
