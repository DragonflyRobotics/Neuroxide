#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <thread>

static cudaStream_t stream;
class MemoryPool {
private:
  void *start;             // starting offset for pool
  float *current_position; // current position in pool
  size_t pool_size;        // pool size in bytes

public:
  MemoryPool(size_t max_size_bytes) {
    cudaMallocAsync(
        &start, max_size_bytes,
        stream); // allocate pool and put starting pose in the float*
    current_position = (float *)
        start; // set the current pose to start since nothing is malloced yet
    pool_size = max_size_bytes;
    printf("Pool initialized with %zu bytes", max_size_bytes);
  }

  MemoryPool() {
    size_t mf, ma;
    cudaMemGetInfo(&mf, &ma);
    printf("Allocating max allowed size of %zu out of %zu...\n", mf, ma);
    size_t max_size_bytes = mf - (1L << 30);
    cudaMallocAsync(
        &start, max_size_bytes,
        stream); // allocate pool and put starting pose in the float*
    current_position = (float *)
        start; // set the current pose to start since nothing is malloced yet
    pool_size = max_size_bytes;
    printf("Pool initialized with %zu bytes\n", max_size_bytes);
  }

  ~MemoryPool() { destroy(); }

  cudaStream_t get_stream() { return stream; }

  float *malloc(size_t bytes, size_t alignment = 128) {
    uintptr_t current_addr = reinterpret_cast<uintptr_t>(current_position);

    // Align to next multiple of `alignment`
    uintptr_t aligned_addr = (current_addr + alignment - 1) & ~(alignment - 1);
    float *aligned_ptr = reinterpret_cast<float *>(aligned_addr);

    // Calculate how much memory was used (includes padding)
    size_t total_bytes = (aligned_addr - current_addr) + bytes;
    size_t advance = (total_bytes + sizeof(float) - 1) / sizeof(float);

    float *new_position = current_position + advance;

    if ((char *)new_position > (char *)start + pool_size) {
      printf("Not enough space in pool! Requested %zu aligned bytes\n", bytes);
      exit(4);
    }

    current_position = new_position;
    return aligned_ptr;
  }

  void reset() {
    destroy();

    cudaMallocAsync(
        &start, pool_size,
        stream); // allocate pool and put starting pose in the float*
    current_position = (float *)
        start; // set the current pose to start since nothing is malloced yet
    printf("Pool initialized with %zu bytes\n", pool_size);
  }

  void destroy() {
    cudaFree(start);
    printf("Pool Destroyed\n");
  }
};

int main() {
  MemoryPool pool = MemoryPool();
  pool.malloc(256);
  pool.malloc(256);
  pool.malloc(256);
  pool.malloc(256);
  std::this_thread::sleep_for(std::chrono::seconds(2));
  pool.destroy();
  return 0;
}
