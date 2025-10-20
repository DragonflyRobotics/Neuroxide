use crate::types::{device::Device, tensor_err::TensorCreationError};

pub struct TensorData<T> {
    device: Device,
    shape: Box<[usize]>,
    values: Vec<T>,
    cuda_ptr: Option<*mut f32>,
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

    pub fn get_device(&self) -> &Device {
        &self.device
    }

    pub fn get_shape(&self) -> &Box<[usize]> {
        &self.shape
    }

    pub fn get_values(&self) -> &Vec<T> {
        &self.values
    }
}
