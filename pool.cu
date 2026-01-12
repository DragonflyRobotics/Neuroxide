#include <cstddef>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>

static cudaStream_t stream;
class Block {
    Block *prev;
    Block *next;
    size_t size_bytes;
    bool is_free;
    const void *start;

  public:
    Block(size_t initial_size, const void *start_ptr)
        : prev(nullptr), next(nullptr), size_bytes(initial_size), is_free(true),
          start(start_ptr) {}
    size_t get_size_bytes() const { return size_bytes; }
    bool get_is_free() const { return is_free; }
    const void *get_start() const { return start; }
    void set_is_free(bool free) { is_free = free; }
    void set_next(Block *n) { next = n; }
    Block *get_next() const { return next; }
    Block *get_prev() const { return prev; }
    void set_prev(Block *p) { prev = p; }
    void set_size_bytes(size_t s) { size_bytes = s; }
};

class Segment {
    size_t size_bytes;
    Block *block;

  public:
    Segment(size_t size, Block *b) : size_bytes(size), block(b) {}
    size_t get_size_bytes() const { return size_bytes; }
    Block *get_block() const { return block; }
};

class Tensor {
    size_t size_bytes;
    size_t dtype_size_bytes;
    Block *block;

  public:
    Tensor(size_t size, size_t dtype_size)
        : size_bytes(size * dtype_size), dtype_size_bytes(dtype_size),
          block(nullptr) {}
    size_t get_size_bytes() const { return size_bytes; }
    Block *get_block() const { return block; }
    size_t get_dtype_size_bytes() const { return dtype_size_bytes; }
    Block *set_block(Block *b) {
        if (b->get_is_free() == false) {
            throw std::runtime_error("Block is already allocated");
        }
        block = b;
        block->set_is_free(false);
        if (this->size_bytes > b->get_size_bytes()) {
            throw std::runtime_error("Block too small for tensor allocation");
        }
        if (this->size_bytes < b->get_size_bytes()) {
            // if (b->get_next() != nullptr) {
            //     throw std::runtime_error(
            //         "Block splitting only supported for last block in
            //         segment");
            // }
            // optionally split block here
            Block *new_block =
                new Block(b->get_size_bytes() - this->size_bytes,
                          static_cast<const void *>(
                              static_cast<const char *>(b->get_start()) +
                              this->size_bytes));
            new_block->set_prev(b);
            if (b->get_next() != nullptr) {
                new_block->set_next(b->get_next());
                b->get_next()->set_prev(new_block);
            }
            b->set_next(new_block);
            b->set_size_bytes(this->size_bytes);
            std::cout << "Splitting block: allocated " << this->size_bytes
                      << " bytes, created new block of "
                      << new_block->get_size_bytes() << " bytes\n";
            return new_block;
        }
        return nullptr;
    }
};

struct SizeCompare {
    using is_transparent = void; // enables heterogeneous lookup
    bool operator()(const Block *a, const Block *b) const {
        return a->get_size_bytes() < b->get_size_bytes(); // sort ascending
    }

    bool operator()(size_t size, const Block *b) const {
        return size < b->get_size_bytes();
    }

    bool operator()(const Block *a, size_t size) const {
        return a->get_size_bytes() < size;
    }
};

class Allocator {
    std::vector<Segment> segments;
    std::multiset<Block *, SizeCompare> free_blocks;
    const size_t BLOCK_SIZE = 1 << 26; // 64MB

  public:
    void malloc(Tensor &tensor) {
        auto it = free_blocks.lower_bound(tensor.get_size_bytes());
        Block *suitable_block = it != free_blocks.end() ? *it : nullptr;
        if (suitable_block == nullptr) {
            suitable_block = new_segment(tensor.get_size_bytes());
        }
        auto it2 = free_blocks.find(suitable_block);
        if (it2 != free_blocks.end()) {
            free_blocks.erase(it2);
        }
        // allocate tensor from suitable_block
        Block *next_block = tensor.set_block(suitable_block);
        if (next_block != nullptr) {
            free_blocks.insert(next_block);
        }

        // cleanly print segment and blocks info
        print_segments_and_blocks();
    }

    void free(Tensor &tensor) {
        Block *block = tensor.get_block();
        if (block == nullptr) {
            throw std::runtime_error("Tensor has no associated block to free");
        }
        if (block->get_is_free()) {
            throw std::runtime_error("Double free detected");
        }
        block->set_is_free(true);
        if (block->get_next() != nullptr && block->get_next()->get_is_free()) {
            // merge with next block
            Block *next_block = block->get_next();
            block->set_size_bytes(block->get_size_bytes() +
                                  next_block->get_size_bytes());
            block->set_next(next_block->get_next());
            next_block->set_prev(block);
            auto it = free_blocks.find(next_block);
            if (it != free_blocks.end()) {
                free_blocks.erase(it);
            }
            delete next_block;
            std::cout << "Merging with next free block\n";
        }
        if (block->get_prev() != nullptr && block->get_prev()->get_is_free()) {
            // merge with previous block
            Block *prev_block = block->get_prev();
            prev_block->set_size_bytes(prev_block->get_size_bytes() +
                                       block->get_size_bytes());
            prev_block->set_next(block->get_next());
            auto it = free_blocks.find(block);
            if (it != free_blocks.end()) {
                free_blocks.erase(it);
            }
            delete block;
            block = prev_block;
            std::cout << "Merging with previous free block\n";
        }
        free_blocks.insert(block);
        print_segments_and_blocks();
    }

    Block *new_segment(size_t size_bytes) {
        // create new segment and add to segments
        size_t to_alloc = std::max(size_bytes, BLOCK_SIZE);
        void *segment_start;
        cudaMallocAsync(&segment_start, to_alloc, stream);
        // create initial free block covering entire segment
        Block *initial_block = new Block(to_alloc, segment_start);
        free_blocks.insert(initial_block);
        segments.emplace_back(to_alloc, initial_block);
        return initial_block;
    }

    void print_segments_and_blocks() const {
        std::cout << "=== Allocator State ===\n";
        for (size_t i = 0; i < segments.size(); ++i) {
            const Segment &seg = segments[i];
            std::cout << "Segment " << i << ": " << seg.get_size_bytes()
                      << " bytes\n";
            // Traverse blocks in this segment
            const Block *block = seg.get_block();
            int block_idx = 0;
            while (block) {
                std::cout << "  Block " << block_idx
                          << ": start=" << block->get_start()
                          << ", size=" << block->get_size_bytes() / (1e6)
                          << " MB, "
                          << (block->get_is_free() ? "FREE" : "ALLOCATED")
                          << "\n";
                block = block->get_next();
                ++block_idx;
            }
        }
        std::cout << "=======================\n";
    }
};

int main() {
    Allocator allocator;
    Tensor t1(1, sizeof(float));
    Tensor t2(1, sizeof(float));
    Tensor t3(1, sizeof(float));
    Tensor t4(1, sizeof(float));
    Tensor t5(1024 * 1024, sizeof(float));
    Tensor t6(1024 * 1024, sizeof(float));
    Tensor t7(1024 * 1024, sizeof(float));
    Tensor t8(5024 * 5024, sizeof(float));
    Tensor t9(1024 * 1024, sizeof(float));
    allocator.malloc(t1);
    allocator.malloc(t2);
    allocator.malloc(t3);
    allocator.malloc(t4);
    allocator.malloc(t5);
    allocator.malloc(t6);
    allocator.malloc(t7);
    allocator.malloc(t8);
    allocator.malloc(t9);

    allocator.free(t2);
    allocator.malloc(t2);
    allocator.free(t2);
    allocator.free(t3);
    allocator.free(t1);
    Tensor t10(3, sizeof(float));
    allocator.malloc(t10);

    // sleep for a while to let cudaMallocAsync complete
    cudaDeviceSynchronize();
}
