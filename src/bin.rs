#![allow(warnings)]
use std::ptr;
use std::ffi::{c_int, c_void};

// Declare cuDNN API functions
#[link(name = "cudart")]
extern "C" {
fn cudaGetDeviceCount(count: *mut i32) -> cudnnStatus_t;
fn cudaMalloc(ptr: *mut *mut c_void, size: usize) -> c_int;
fn cudaMemcpy(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: c_int,
) -> c_int;
fn cudaFree(ptr: *mut c_void) -> c_int;
}

#[link(name = "cudnn_ops")]
extern "C" {
pub fn cudnnCreateTensorDescriptor(tensorDesc: *mut cudnnTensorDescriptor_t) -> c_int;
pub fn cudnnSetTensor4dDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    format: c_int,
    dataType: c_int,
    n: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> c_int;
pub fn cudnnAddTensor(
    handle: cudnnHandle_t,
    alpha: *const f32,
    a_desc: cudnnTensorDescriptor_t,
    A: *const f32,
    beta: *const f32,
    c_desc: cudnnTensorDescriptor_t,
    C: *mut f32,
) -> c_int;
pub fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> c_int;
}

#[link(name = "cudnn")]
extern "C" {
fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;
fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;
}

#[link(name = "math_kernel", kind = "static")]
extern "C" {
fn vectorAdd_main();
}

// cuDNN types
pub type cudnnHandle_t = *mut std::ffi::c_void;
pub type cudnnStatus_t = i32; // usually cuDNN uses enums as return statuses

pub type cudnnTensorDescriptor_t = *mut c_void;

// Define cuDNN constants
pub const CUDNN_STATUS_SUCCESS: c_int = 0;
pub const CUDNN_DATA_FLOAT: c_int = 0;
pub const CUDNN_TENSOR_NCHW: c_int = 0;
pub const cudaMemcpyHostToDevice: c_int = 1;
pub const cudaMemcpyDeviceToHost: c_int = 2;

fn main() {
    // Initialize cuDNN
    let mut handle: cudnnHandle_t = std::ptr::null_mut();

    unsafe {
        let status = cudnnCreate(&mut handle);
        if status != 0 {
            panic!("cuDNN initialization failed: {}", status);
        }
        // Do cuDNN operations here
        let mut count: i32 = 0;
        let status = cudaGetDeviceCount(&mut count);
        if status != 0 {
            panic!("cuDNN device count failed: {}", status);
        }


        // // Create a tensor descriptor
        // let mut tensor_desc: cudnnTensorDescriptor_t = ptr::null_mut();
        // let status = cudnnCreateTensorDescriptor(&mut tensor_desc);
        // if status != CUDNN_STATUS_SUCCESS {
        //     panic!("Failed to create tensor descriptor: {:?}", status);
        // }
        //
        // // Set the tensor descriptor
        // let n = 1; // Batch size
        // let c = 1; // Number of channels
        // let h = 2; // Height
        // let w = 2; // Width
        // let data_type = CUDNN_DATA_FLOAT; // Data type
        // let format = CUDNN_TENSOR_NCHW; // Tensor format
        // //
        //
        // let status = cudnnSetTensor4dDescriptor(
        //     tensor_desc,
        //     format,
        //     data_type,
        //     n,
        //     c,
        //     h,
        //     w,
        // );
        //
        //
        // // Allocate memory on GPU for input tensors and output tensor
        // let size = (n * c * h * w) as usize * std::mem::size_of::<f32>();
        // let mut d_A: *mut c_void = ptr::null_mut();
        // let mut d_B: *mut c_void = ptr::null_mut();
        // let mut d_C: *mut c_void = ptr::null_mut();
        //
        // let status = cudaMalloc(&mut d_A, size);
        // if status != 0 {
        //     panic!("Failed to allocate device memory for A: {}", status);
        // }
        //
        // let status = cudaMalloc(&mut d_B, size);
        // if status != 0 {
        //     panic!("Failed to allocate device memory for B: {}", status);
        // }
        //
        // let status = cudaMalloc(&mut d_C, size);
        // if status != 0 {
        //     panic!("Failed to allocate device memory for C: {}", status);
        // }
        //
        // // Initialize input tensors on the host
        // let h_A: [f32; 4] = [1.0, 2.0, 3.0, 4.0];
        // let h_B: [f32; 4] = [5.0, 6.0, 7.0, 8.0];
        //
        // for _ in 0..500000 {
        //     // Copy input tensors from host to device
        //     let status = cudaMemcpy(d_A, h_A.as_ptr() as *const c_void, size, cudaMemcpyHostToDevice);
        //     if status != 0 {
        //         panic!("Failed to copy data from host to device A: {}", status);
        //     }
        //
        //     let status = cudaMemcpy(d_B, h_B.as_ptr() as *const c_void, size, cudaMemcpyHostToDevice);
        //     if status != 0 {
        //         panic!("Failed to copy data from host to device B: {}", status);
        //     }
        //
        // }
        //
        // // Perform addition
        // let alpha: f32 = 1.0;
        // let beta: f32 = 1.0;
        //
        // let status = cudnnAddTensor(
        //     handle,
        //     &alpha,
        //     tensor_desc,
        //     d_A as *const f32,
        //     &beta,
        //     tensor_desc,
        //     d_B as *mut f32,
        // );
        //
        // if status != CUDNN_STATUS_SUCCESS {
        //     panic!("Failed to perform tensor addition: {:?}", status);
        // }
        //
        // // Allocate space for the result tensor on the host
        // let mut h_C: [f32; 4] = [0.0; 4];
        //
        // // Copy the result tensor from device to host
        // let status = cudaMemcpy(h_C.as_mut_ptr() as *mut c_void, d_B, size, cudaMemcpyDeviceToHost);
        // if status != 0 {
        //     panic!("Failed to copy data from device to host C: {}", status);
        // }
        //
        // // Print the result
        // println!("Result tensor C:");
        // for value in &h_C {
        //     print!("{} ", value);
        // }
        // println!();
        //
        // // Free device memory
        // cudaFree(d_A);
        // cudaFree(d_B);
        // cudaFree(d_C);
        //
        // if status != CUDNN_STATUS_SUCCESS {
        //     panic!("Failed to set tensor descriptor: {:?}", status);
        // }

        println!("Device count: {}", count);



        vectorAdd_main();
        println!("Hello, world!");

        // // Destroy cuDNN handle
        cudnnDestroy(handle);
    }
}

