extern crate cc;

#[cfg(feature = "cuda")] fn main() { use std::env; use std::path::PathBuf; use std::fs; // Check for the CUDA toolkit installation path
    let current_dir = env::current_dir().expect("Failed to get current dir");
    println!("cargo:warning=Current directory: {}", current_dir.display());

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
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-lib=cutensor");

    println!("cargo:rustc-link-lib=dylib=stdc++");


    // TODO: Fix this to be solid later
    let cuda_dir = "cuda"; // Replace with your actual directory if different

    // Compile all CUDA files in the directory
    if let Ok(entries) = fs::read_dir(cuda_dir) {
        for entry in entries.filter_map(Result::ok) {
            if let Some(extension) = entry.path().extension() {
                if extension == "cu" {
                    let mut build = cc::Build::new();
                    build.compiler("clang++"); // Use clang++ for CUDA compilation
                    // build.flag("--cuda-gpu-arch=sm_120");
                    // build.flag("--cuda-gpu-arch=sm_100");
                    // build.flag("--cuda-gpu-arch=sm_90");
                    build.flag("--cuda-gpu-arch=sm_89");
                    // build.flag("--cuda-gpu-arch=sm_87");
                    // build.flag("--cuda-gpu-arch=sm_86");
                    // build.flag("--cuda-gpu-arch=sm_80");
                    // build.flag("--cuda-gpu-arch=sm_75");
                    build.flag("-O3"); // or any standard you want

                    print!("Compiling CUDA file: {}", entry.path().display());

                    build.file(entry.path())
                        .compile(format!("lib{}.a", entry.path().file_stem().unwrap().to_str().unwrap()).as_str());
                }
            }
        }
    } else {
        eprintln!("Error: Could not read directory {}", cuda_dir);
    }
    println!("cargo::rerun-if-changed=build.rs");
    println!("cargo::rerun-if-changed={}", cuda_dir);
    println!("cargo:rustc-check-cfg=cfg(tarpaulin_include)");
    
    // OpenBLAS
    println!("cargo:rustc-link-search=native=/usr/lib64"); // Path to the OpenBLAS library
    println!("cargo:rustc-link-lib=dylib=openblas");      // Link with the OpenBLAS dynamic library
}

#[cfg(not(feature = "cuda"))]
fn main() {
    // Do nothing
    // link openblas
    println!("cargo:rustc-link-search=native=/usr/lib64"); // Path to the OpenBLAS library
    println!("cargo:rustc-link-lib=dylib=openblas");      // Link with the OpenBLAS dynamic library
}
