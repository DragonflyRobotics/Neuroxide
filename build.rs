extern crate cc;
use std::env;
use std::path::PathBuf;

fn main() {
    // Check for the CUDA toolkit installation path
    let cuda_path = match env::var("CUDA_PATH") {
        Ok(path) => PathBuf::from(path),
        Err(_) => PathBuf::from("/usr/local/cuda"), // fallback path for Linux
    };

    // Specify the CUDA library path to the linker
    println!(
        "cargo:rustc-link-search=native={}",
        cuda_path.join("lib64").display()
    );

    // Link against the cudart library
    println!("cargo:rustc-link-lib=cudart");

    cc::Build::new()
        .cuda(true)
        .flag("-cudart=shared")
        .flag("-gencode")
        .flag("arch=compute_75,code=sm_75")
        .file("cuda/math_kernel.cu")
        .compile("libmath_kernel.a");

}

