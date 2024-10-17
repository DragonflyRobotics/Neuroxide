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
    }
    mem
}
