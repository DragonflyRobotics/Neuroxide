use crate::pointers::{DevicePtr, Pointer};

pub struct DummyPtr<T> {
    pub ptr: *mut T,
}

impl<T: 'static> Pointer<T> for DummyPtr<T> {
    fn new(ptr: *mut T) -> Box<dyn DevicePtr> {
        Box::new(DummyPtr { ptr })
    }

    fn shift(&self, offset: isize) -> DummyPtr<T> {
        unsafe {
            DummyPtr {
                ptr: self.ptr.offset(offset),
            }
        }
    }
}

impl<T: 'static> DevicePtr for DummyPtr<T> {
    fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr as *mut std::ffi::c_void
    }

    fn size_of_elem(&self) -> usize {
        std::mem::size_of::<T>()
    }

    fn clone(&self) -> Box<dyn DevicePtr> {
        DummyPtr::new(self.ptr)
    }

    fn offset(&self, offset: usize) -> Box<dyn DevicePtr> {
        Box::new(self.shift(offset as isize))
    }
}
