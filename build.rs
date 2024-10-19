extern crate cc;

#[cfg(feature = "cuda")]
fn main() {
    use std::env;
    use std::path::PathBuf;
    use std::fs;
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


    let cuda_dir = "cuda"; // Replace with your actual directory if different

    // Compile all CUDA files in the directory
    if let Ok(entries) = fs::read_dir(cuda_dir) {
        for entry in entries.filter_map(Result::ok) {
            if let Some(extension) = entry.path().extension() {
                if extension == "cu" {
                    let mut build = cc::Build::new();
                    build.cuda(true);
                    build.flag("-cudart=shared");

                    // Add each gencode specification individually
                    let gencode_flags = [
                        "arch=compute_50,code=sm_50",
                        "arch=compute_60,code=sm_60",
                        "arch=compute_61,code=sm_61",
                        "arch=compute_70,code=sm_70",
                        "arch=compute_75,code=sm_75",
                        "arch=compute_80,code=sm_80",
                    ];

                    for flag in &gencode_flags {
                        build.flag("-gencode").flag(flag);
                    }

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
}

#[cfg(not(feature = "cuda"))]
fn main() {
    // Do nothing
}
