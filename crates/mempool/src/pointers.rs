use std::ops::{Add, Sub};

pub trait Pointer<T: 'static> {
    fn new(ptr: *mut T) -> Box<dyn DevicePtr>;
    fn shift(&self, offset: isize) -> Self;
}

pub trait DevicePtr {
    fn as_ptr(&self) -> *mut std::ffi::c_void; // opaque pointer
    fn size_of_elem(&self) -> usize;
    fn clone(&self) -> Box<dyn DevicePtr>;
    fn offset(&self, offset: usize) -> Box<dyn DevicePtr>;
}
impl Clone for Box<dyn DevicePtr> {
    fn clone(&self) -> Box<dyn DevicePtr> {
        self.as_ref().clone()
    }
}

impl Add<usize> for Box<dyn DevicePtr> {
    type Output = Box<dyn DevicePtr>;

    fn add(self, offset: usize) -> Self::Output {
        self.offset(offset)
    }
}

impl Sub<usize> for Box<dyn DevicePtr> {
    type Output = Box<dyn DevicePtr>;

    fn sub(self, offset: usize) -> Self::Output {
        self.offset(-(offset as isize) as usize)
    }
}
