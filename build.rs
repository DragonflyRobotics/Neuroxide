extern crate cc;
use std::env;
use std::path::PathBuf;
use std::fs;

#[cfg(feature = "cuda")]
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


    let cuda_dir = "cuda"; // Replace with your actual directory if different

    // Compile all CUDA files in the directory
    if let Ok(entries) = fs::read_dir(cuda_dir) {
        for entry in entries.filter_map(Result::ok) {
            if let Some(extension) = entry.path().extension() {
                if extension == "cu" {
                    // Print the new file name
                    cc::Build::new()
                        .cuda(true)
                        .flag("-cudart=shared")
                        .flag("-gencode")
                        .flag("arch=compute_75,code=sm_75")
                        .file(entry.path())
                        .compile(format!("lib{}.a", entry.path().file_stem().unwrap().to_str().unwrap()).as_str());
                    // println!("Compiling: {:?} ==> {:?}", entry.path(), format!("lib{}.a", entry.path().file_stem().unwrap().to_str().unwrap()));
                }
            }
        }
    } else {
        eprintln!("Error: Could not read directory {}", cuda_dir);
    }

}

#[cfg(not(feature = "cuda"))]
fn main() {
    // Do nothing
}
