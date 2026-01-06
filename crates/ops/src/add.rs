use std::sync::{Arc, Mutex};

use rayon::prelude::*;
use types::input::ToTensorInputs;
use types::op_stub::OperationStub;
use types::tensor::Tensor;
use types::tensor_element::{SharedTensor, TensorElement, TensorHandleExt};

use crate::mul::Mul;
use crate::op::Operation;

#[derive(Debug)]
pub struct Add<T> {
    input_tensors: Box<[SharedTensor<T>]>,
}

impl<T: TensorElement> OperationStub<T> for Add<T> {
    fn check_forward_inputs<I: ToTensorInputs<T>>(
        inputs: I,
    ) -> Result<Box<[SharedTensor<T>]>, String>
    where
        Self: Sized,
    {
        let inputs = inputs.into_inputs();
        if inputs.len() != 2 {
            return Err("Add operation requires exactly two input tensors.".to_string());
        }
        Ok(inputs)
    }
    fn forward_cpu(inputs: Box<[SharedTensor<T>]>) -> SharedTensor<T> {
        let mut a = inputs[0].clone();
        let mut b = inputs[1].clone();

        let result_shape;
        let mut result_values;
        if Arc::ptr_eq(&inputs[0], &inputs[1]) {
            result_shape = a.get_shape().clone();
            result_values = vec![T::from(0).unwrap(); result_shape.iter().product()];
            let a_lock = a.lock_ref();
            let a_values = a_lock.get_values();
            if result_values.len() > 1024 {
                result_values.par_iter_mut().enumerate().for_each(|(i, o)| {
                    o.clone_from(&(a_values[i] + a_values[i]));
                });
            } else {
                result_values.iter_mut().enumerate().for_each(|(i, o)| {
                    o.clone_from(&(a_values[i] + a_values[i]));
                });
            }
        } else {
            (a, b) = Tensor::broadcast_linear(&inputs[0], &inputs[1]);
            result_shape = a.get_shape().clone();
            result_values = vec![T::from(0).unwrap(); result_shape.iter().product()];
            let a_lock = a.lock_ref();
            let b_lock = b.lock_ref();
            let a_values = a_lock.get_values_slice();
            let b_values = b_lock.get_values_slice();
            if result_values.len() > 1024 {
                result_values.par_iter_mut().enumerate().for_each(|(i, o)| {
                    o.clone_from(&(a_values[i] + b_values[i]));
                });
            } else {
                result_values.par_iter_mut().enumerate().for_each(|(i, o)| {
                    o.clone_from(&(a_values[i] + b_values[i]));
                });
            }
        };
        let result_tensor = Tensor::new(result_values, result_shape);
        let add = Add {
            input_tensors: Box::new([a, b]),
        };
        result_tensor
            .lock()
            .unwrap()
            .set_op(Arc::new(Mutex::new(add)));
        result_tensor
    }

    fn forward_cuda(_inputs: Box<[SharedTensor<T>]>) -> SharedTensor<T> {
        todo!("CUDA not implemented for Add operation");
    }

    fn backward(&mut self, upstream: SharedTensor<T>) {
        // Backward implementation for Add operation
        if Arc::ptr_eq(&self.input_tensors[0], &self.input_tensors[1]) {
            let op_grad = {
                let a = self.input_tensors[0].lock().unwrap();
                Mul::forward((
                    upstream.clone(),
                    Tensor::new(
                        vec![T::from(2).unwrap(); a.get_values().len()],
                        a.get_shape().clone(),
                    ),
                ))
            };
            Self::apply_grad(self.input_tensors[0].clone(), op_grad.clone());
            Self::recurse_backward(self.input_tensors[0].clone());
        } else {
            let op_grad = Tensor::new(
                upstream.lock().unwrap().get_values().clone(),
                self.input_tensors[0].lock().unwrap().get_shape().clone(),
            );
            Self::apply_grad(self.input_tensors[0].clone(), op_grad.clone());
            Self::apply_grad(self.input_tensors[1].clone(), op_grad.clone());
            Self::recurse_backward(self.input_tensors[0].clone());
            Self::recurse_backward(self.input_tensors[1].clone());
        };
    }

    fn get_branches(&self) -> Box<[SharedTensor<T>]> {
        self.input_tensors.clone()
    }
}
