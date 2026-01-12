use crate::device_allocator::DeviceAllocator;
use crate::dummyptr::DummyPtr;
use crate::pointers::{DevicePtr, Pointer};

pub struct DummyDevice {
    offset: usize,
}

impl DummyDevice {
    pub fn malloc(&mut self, size: usize) -> *mut u8 {
        let start = self.offset;
        self.offset += size;
        start as *mut u8
    }

    pub fn free(&mut self, _ptr: *mut u8) {
        // No-op for dummy device
    }
}

impl DeviceAllocator for DummyDevice {
    fn new() -> Box<dyn DeviceAllocator> {
        Box::new(DummyDevice { offset: 0 })
    }
    fn allocate(&mut self, size: usize) -> Result<Box<dyn DevicePtr>, String> {
        println!("Allocating {} bytes on CUDA device", size);
        Ok(DummyPtr::new(self.malloc(size)))
    }

    fn deallocate(&mut self, ptr: Box<dyn DevicePtr>) {
        println!("Deallocating memory on CUDA device");
        self.free(ptr.as_ptr() as *mut u8);
    }
}
