use std::time::Instant;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use neuroxide::mempool::pool::PoolTrait;
use neuroxide::mempool::pool::Pool;
use neuroxide::cuda::memory::cudadevice::CudaDevice; // or whatever your real device allocator is
use neuroxide::mempool::device_allocator::DeviceAllocator;

fn bench_pool_alloc_free_stress() {
    let device = CudaDevice::new(); // or DummyDevice if you want CPU-only first
    let mut pool = Pool::new(device);

    let iters = 50_000;
    let min_size = 1 * 1024;        // 1 KB
    let max_size = 128 * 1024 * 1024; // 512 MB
    let keep_prob = 0.1;

    let mut rng = StdRng::seed_from_u64(12345);
    let mut live: Vec<_> = Vec::new();

    // Warmup
    for _ in 0..1000 {
        let b = pool.malloc(4096).unwrap();
        pool.free(&b);
    }

    let start = Instant::now();

    for i in 0..iters {
        let size = rng.gen_range(min_size..=max_size);
        let block = pool.malloc(size).expect("alloc failed");

        if rng.r#gen::<f32>() < keep_prob {
            live.push(block);
        } else {
            pool.free(&block);
        }

        // Occasionally free a random live block (fragmentation stress)
        if !live.is_empty() && rng.r#gen::<f32>() < 0.1 {
            let idx = rng.gen_range(0..live.len());
            let b = live.swap_remove(idx);
            pool.free(&b);
        }

        // Optional: print progress
        if (i + 1) % 10_000 == 0 {
            println!("iter {}", i + 1);
        }
    }

    // Cleanup
    for b in live {
        pool.free(&b);
    }

    let dt = start.elapsed().as_secs_f64();
    println!("=== Rust Pool benchmark ===");
    println!("Iterations: {}", iters);
    println!("Time: {:.3} s", dt);
    println!("Ops/sec: {:.1}", iters as f64 / dt);
}

fn main() {
    let iters = 50_000;
    let min_size = 1 * 1024;        // 1 KB
    let max_size = 512 * 1024 * 1024; // 512 MB
    let keep_prob = 0.3;

    bench_pool_alloc_free_stress();
}