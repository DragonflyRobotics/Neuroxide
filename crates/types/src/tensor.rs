use std::sync::{Arc, Mutex};

use crate::{
    cat::Cat,
    device::Device,
    input::{ToShapeInputs, ToTensorValueInputs},
    op_stub::OperationStub,
    tensor_data::TensorData,
    tensor_element::{SharedTensor, TensorElement, TensorHandleExt},
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

    pub fn get_shape(&self) -> Box<[usize]> {
        self.data.shape.clone()
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

    pub fn get_stride(&self) -> Box<[usize]> {
        let mut stride = vec![1; self.data.shape.len()];
        for i in (0..self.data.shape.len() - 1).rev() {
            stride[i] = stride[i + 1] * self.data.shape[i + 1];
        }
        stride.into_boxed_slice()
    }

    pub fn cat(one: &SharedTensor<T>, other: &SharedTensor<T>, axis: usize) -> SharedTensor<T> {
        if one.get_shape().len() != other.get_shape().len() {
            panic!("Tensors must have the same number of dimensions to concatenate.");
        }

        let mut outer_dims = 1;
        let mut inner_dims = 1;
        for i in 0..one.get_shape().len() {
            if i != axis && one.get_shape()[i] != other.get_shape()[i] {
                panic!("Tensors must have the same shape except along the concatenation axis.");
            }
            if i < axis {
                outer_dims *= one.get_shape()[i];
            }
            if i > axis {
                inner_dims *= one.get_shape()[i];
            }
        }

        let self_slab_size = one.get_shape()[axis] * inner_dims;
        let other_slab_size = other.get_shape()[axis] * inner_dims;

        let mut final_shape = one.get_shape().clone();
        final_shape[axis] += other.get_shape()[axis];

        let mut final_vector: Vec<T> = Vec::with_capacity(final_shape.iter().product());
        unsafe {
            final_vector.set_len(final_shape.iter().product());
        }

        for i in 0..outer_dims {
            let mut starta = 0;
            let mut startb = 0;
            let mut residual = i;
            for d in 0..axis {
                let dim_size = one.get_shape()[d];
                let idx = residual % dim_size; // index along this dim
                residual /= dim_size; // update residual for next dim
                starta += idx * one.get_stride()[d];
                startb += idx * other.get_stride()[d];
            }

            let out_base = i * (self_slab_size + other_slab_size);
            final_vector[out_base..out_base + self_slab_size]
                .copy_from_slice(&one.values()[starta..starta + self_slab_size]);

            final_vector[out_base + self_slab_size..out_base + self_slab_size + other_slab_size]
                .copy_from_slice(&other.values()[startb..startb + other_slab_size]);
        }

        let data = TensorData::new(final_vector.clone(), final_shape.clone(), Device::CPU).unwrap();
        let op: Option<Arc<Mutex<dyn OperationStub<T>>>> = Some(Arc::new(Mutex::new(Cat {
            input_tensors: Box::new([one.clone(), other.clone()]),
            axis,
        })));
        Arc::new(Mutex::new(Tensor {
            data,
            requires_grad: false,
            gradient: None,
            op,
        }))
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
