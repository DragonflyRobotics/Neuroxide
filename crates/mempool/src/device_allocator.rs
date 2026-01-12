use crate::pointers::DevicePtr;

pub trait DeviceAllocator {
    fn new() -> Box<dyn DeviceAllocator>
    where
        Self: Sized;
    fn allocate(&mut self, size: usize) -> Result<Box<dyn DevicePtr>, String>;
    fn deallocate(&mut self, ptr: Box<dyn DevicePtr>);
}
