use mempool::pointers::{DevicePtr, Pointer};

pub struct CudaPtr<T> {
    pub ptr: *mut T,
}

impl<T: 'static> Pointer<T> for CudaPtr<T> {
    fn new(ptr: *mut T) -> Box<dyn DevicePtr> {
        return Box::new(CudaPtr { ptr });
    }

    fn shift(&self, offset: isize) -> CudaPtr<T> {
        unsafe {
            CudaPtr {
                ptr: self.ptr.offset(offset),
            }
        }
    }
}

impl<T: 'static> DevicePtr for CudaPtr<T> {
    fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.ptr as *mut std::ffi::c_void
    }

    fn size_of_elem(&self) -> usize {
        std::mem::size_of::<T>()
    }

    fn clone(&self) -> Box<dyn DevicePtr> {
        CudaPtr::new(self.ptr)
    }

    fn offset(&self, offset: usize) -> Box<dyn DevicePtr> {
        Box::new(self.shift(offset as isize))
    }
}
