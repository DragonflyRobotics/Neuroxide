use std::process::Command;
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

    // Path to the CUDA file
    let cuda_file = "cuda/math_kernel.cu";

    // Compile the CUDA file using nvcc
    let status = Command::new("nvcc")
        .args(&["-c", cuda_file, "-o", "math_kernel.o"]) // Generate an object file
        .status()
        .expect("Failed to compile CUDA code");

    if !status.success() {
        panic!("CUDA compilation failed!");
    }

    // Tell Cargo to include the compiled object file
    println!("cargo:rustc-link-lib=math_kernel");
    println!("cargo:rustc-link-search=native=.");

}

