fn main() {
    // Set the paths to your CUDA and cuDNN installation directories
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64"); // Adjust the path as necessary
    println!("cargo:rustc-link-search=native=/usr/local/cudnn/lib64"); // Adjust the path as necessary

    // Link against the cuDNN and CUDA libraries
    println!("cargo:rustc-link-lib=dylib=cudnn");
    println!("cargo:rustc-link-lib=dylib=cuda");
}

