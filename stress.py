import torch
import time
import random

device = torch.device("cuda")

def bench_torch_alloc_free_stress(
    iters=50_000,
    min_bytes=1 * 1024,
    max_bytes=128 * 1024 * 1024,
    keep_prob=0.1,
):
    live = []

    # Warmup
    for _ in range(1000):
        x = torch.empty(1024, device=device)
        del x
    torch.cuda.synchronize()

    start = time.perf_counter()

    for i in range(iters):
        size_bytes = random.randint(min_bytes, max_bytes)
        # use uint8 so bytes = numel
        x = torch.empty(size_bytes, device=device, dtype=torch.float)

        if random.random() < keep_prob:
            live.append(x)
        else:
            del x

        if live and random.random() < 0.1:
            idx = random.randrange(len(live))
            del live[idx]

        if (i + 1) % 10_000 == 0:
            print(f"iter {i+1}")
        torch.cuda.synchronize()

    # Cleanup
    live.clear()
    torch.cuda.synchronize()

    end = time.perf_counter()
    dt = end - start

    print("=== PyTorch CUDA allocator benchmark ===")
    print(f"Iterations: {iters}")
    print(f"Time: {dt:.3f} s")
    print(f"Ops/sec: {iters / dt:.1f}")
    print(torch.cuda.memory_summary())

if __name__ == "__main__":
    bench_torch_alloc_free_stress()
