use neuroxide::utils::cuda;

fn main() {
    let count = cuda::get_cuda_device_count();
    println!("CUDA device count: {}", count);
    let name = cuda::get_cuda_device_name(0);
    println!("CUDA device name: {}", name);

    let mem = cuda::get_cuda_device_mem(0);
    println!("CUDA device memory: {} MB", mem);
}
