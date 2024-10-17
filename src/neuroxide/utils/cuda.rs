use libc::size_t;

#[cfg(feature = "cuda")]
#[link(name = "cudart")]
extern "C" {
fn cudaGetDeviceCount(count: *mut i32) -> CudnnStatusT;
}

#[cfg(feature = "cuda")]
extern "C" {
fn getDeviceName_main(name: *mut u8, device: i32) -> CudnnStatusT;
fn getTotalMem_main(mem: *mut size_t, device: i32) -> CudnnStatusT;
fn vectorAdd_main(len: i32, A: *mut f32, B: *mut f32, C: *mut f32) -> CudnnStatusT;
}


pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses

 
#[cfg(feature = "cuda")]
pub fn get_cuda_device_count() -> i32 {
    let mut count: i32 = 0;
    unsafe {
        cudaGetDeviceCount(&mut count);
    }
    count
}

#[cfg(feature = "cuda")]
pub fn get_cuda_device_name(device: i32) -> String {
    let mut name = [0u8; 256];
    unsafe {
        getDeviceName_main(name.as_mut_ptr(), device);
    }
    std::str::from_utf8(&name).unwrap().to_string()
}

#[cfg(feature = "cuda")]
pub fn get_cuda_device_mem(device: i32) -> size_t {
    let mut mem: size_t = 0;
    unsafe {
        getTotalMem_main(&mut mem, device);
        let mut A = vec![1.0f32, 2.0f32, 3.0f32, 4.0f32];
        let mut B = vec![4.0f32, 3.0f32, 2.0f32, 1.0f32];
        let mut C = vec![0.0f32; 4];
        vectorAdd_main(4, A.as_mut_ptr(), B.as_mut_ptr(), C.as_mut_ptr());
    }
    mem
}
