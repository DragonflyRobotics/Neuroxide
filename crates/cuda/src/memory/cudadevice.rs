use mempool::{device_allocator::DeviceAllocator, pointers::DevicePtr, pointers::Pointer};

use crate::memory::{
    allocate::{allocateDeviceMemory, freeDeviceMemory},
    cudaptr::CudaPtr,
};

pub struct CudaDevice;

impl DeviceAllocator for CudaDevice {
    fn new() -> Box<dyn DeviceAllocator> {
        Box::new(CudaDevice)
    }
    fn allocate(&mut self, size: usize) -> Result<Box<dyn DevicePtr>, String> {
        println!("Allocating {} bytes on CUDA device", size);
        unsafe {
            let ptr = allocateDeviceMemory(size);
            if ptr.is_null() {
                Err("CUDA allocation failed".to_string())
            } else {
                Ok(CudaPtr::new(ptr))
            }
        }
    }

    fn deallocate(&mut self, ptr: Box<dyn DevicePtr>) {
        println!("Deallocating memory on CUDA device");
        unsafe {
            freeDeviceMemory(ptr.as_ptr());
        }
    }
}
