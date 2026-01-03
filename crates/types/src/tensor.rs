use std::sync::{Arc, Mutex};

use crate::{
    device::Device,
    input::{ToShapeInputs, ToTensorValueInputs},
    op_stub::OperationStub,
    tensor_data::TensorData,
    tensor_element::{SharedTensor, TensorElement},
};

pub struct Tensor<T> {
    data: TensorData<T>,
    requires_grad: bool,
    gradient: Option<SharedTensor<T>>,
    op: Option<Arc<Mutex<dyn OperationStub<T>>>>,
}

impl<T: TensorElement> Tensor<T> {
    pub fn new<I: ToShapeInputs, V: ToTensorValueInputs<T>>(data: V, shape: I) -> SharedTensor<T> {
        let data = data.into_values();
        let shape = shape.into_shape();
        let device = Device::CPU;
        let tensor_data = TensorData::new(data, shape, device).unwrap();
        Arc::new(Mutex::new(Tensor {
            data: tensor_data,
            requires_grad: false,
            gradient: None,
            op: None,
        }))
    }

    pub fn ones<I: ToShapeInputs>(shape: I) -> SharedTensor<T> {
        let shape = shape.into_shape();
        let device = Device::CPU;
        let tensor_data = TensorData::new(
            vec![T::from(1).unwrap(); shape.clone().into_iter().product()],
            shape,
            device,
        )
        .unwrap();
        Arc::new(Mutex::new(Tensor {
            data: tensor_data,
            requires_grad: false,
            gradient: None,
            op: None,
        }))
    }

    pub fn set_op(&mut self, op: Arc<Mutex<dyn OperationStub<T>>>) {
        self.op = Some(op);
    }

    pub fn get_values(&self) -> &Vec<T> {
        &self.data.values
    }

    pub fn get_shape(&self) -> Vec<usize> {
        self.data.shape.to_vec().clone()
    }

    pub fn get_op(&self) -> Option<Arc<Mutex<dyn OperationStub<T>>>> {
        self.op.clone()
    }

    pub fn backward(&mut self) {
        self.gradient = Some(Tensor::ones(self.get_shape().clone()));
        let binding = self.op.clone().unwrap();
        let mut op = binding.lock().unwrap();
        // Placeholder for future upstream gradients
        op.backward(self.gradient.clone().unwrap());
    }

    pub fn set_gradient(&mut self, grad: SharedTensor<T>) {
        self.gradient = Some(grad);
    }

    pub fn get_gradient(&self) -> Option<SharedTensor<T>> {
        self.gradient.clone()
    }
}

impl<T> std::fmt::Debug for Tensor<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.data.values)
            .field("shape", &self.data.shape)
            .field("requires_grad", &self.requires_grad)
            .field("op", &self.op.is_some())
            .finish()
    }
}
