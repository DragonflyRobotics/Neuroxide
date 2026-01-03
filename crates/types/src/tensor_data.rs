use crate::{device::Device, tensor_err::TensorCreationError};

pub struct TensorData<T> {
    pub device: Device,
    pub shape: Box<[usize]>,
    pub values: Vec<T>,
    pub cuda_ptr: Option<*mut f32>,
}

impl<T> TensorData<T> {
    pub fn new(
        data: Vec<T>,
        shape: Box<[usize]>,
        device: Device,
    ) -> Result<Self, TensorCreationError> {
        if device == Device::CPU && data.is_empty() {
            return Err(TensorCreationError::InvalidTensorDataCPU);
        }
        if data.len() != shape.iter().product() {
            return Err(TensorCreationError::InvalidTensorDataCPU);
        }
        Ok(TensorData {
            device,
            shape,
            values: data,
            cuda_ptr: None,
        })
    }
}
