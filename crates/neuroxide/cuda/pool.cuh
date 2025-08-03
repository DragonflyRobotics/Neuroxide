#ifndef POOL_CUH
#define POOL_CUH

#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublas_v2.h>


static cudaStream_t stream;
class MemoryPool {
    private:
        void* start; // starting offset for pool
        float* current_position; //current position in pool
        size_t pool_size; //pool size in bytes

    public: 
    MemoryPool(size_t max_size_bytes) {
        cudaMallocAsync(&start, max_size_bytes, stream); // allocate pool and put starting pose in the float*
        current_position = (float*) start; // set the current pose to start since nothing is malloced yet
        pool_size = max_size_bytes;
        printf("Pool initialized with %zu bytes", max_size_bytes);
    }

    MemoryPool() {
        size_t mf, ma;
        cudaMemGetInfo(&mf, &ma);
        printf("Allocating max allowed size of %zu out of %zu...\n", mf, ma);
        size_t max_size_bytes = mf-(1L << 30);
        cudaMallocAsync(&start, max_size_bytes, stream); // allocate pool and put starting pose in the float*
        current_position = (float*) start; // set the current pose to start since nothing is malloced yet
        pool_size = max_size_bytes;
        printf("Pool initialized with %zu bytes\n", max_size_bytes);
    }


    ~MemoryPool() {
        destroy();
    }

    float* malloc (size_t bytes) {
        size_t elements_count = bytes / sizeof(float); // convert bytes to number of floats
        if (current_position + elements_count < (float*)start + (pool_size/sizeof(float))) { // check if there is enough space
            float* return_pos = current_position;
            current_position += elements_count; 
            return return_pos;
        } else {
            printf("Not enough space in pool, trying to allocate %zu bytes\n", bytes);
            exit(4);
        }
    }

    void reset() {
        destroy();

        cudaMallocAsync(&start, pool_size, stream); // allocate pool and put starting pose in the float*
        current_position = (float*) start; // set the current pose to start since nothing is malloced yet
        printf("Pool initialized with %zu bytes\n", pool_size);
    }

    void destroy() {
        cudaFree(start);
        printf("Pool Destroyed\n");
    }
};

inline MemoryPool pool = MemoryPool();


#endif
