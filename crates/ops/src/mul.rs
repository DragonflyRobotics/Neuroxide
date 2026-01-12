use std::sync::{Arc, Mutex};

use crate::op::Operation;
use rayon::iter::IndexedParallelIterator;
use rayon::iter::IntoParallelRefMutIterator;
use rayon::iter::ParallelIterator;
use types::{
    input::ToTensorInputs,
    op_stub::OperationStub,
    tensor::Tensor,
    tensor_element::{SharedTensor, TensorElement, TensorHandleExt},
};

#[derive(Debug)]
pub struct Mul<T> {
    input_tensors: Box<[SharedTensor<T>]>,
}

impl<T: TensorElement> OperationStub<T> for Mul<T> {
    fn check_forward_inputs<I: ToTensorInputs<T>>(
        inputs: I,
    ) -> Result<Box<[SharedTensor<T>]>, String>
    where
        Self: Sized,
    {
        let inputs = inputs.into_inputs();
        if inputs.len() != 2 {
            return Err("Mul operation requires exactly two input tensors.".to_string());
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
                    o.clone_from(&(a_values[i] * a_values[i]));
                });
            } else {
                result_values.iter_mut().enumerate().for_each(|(i, o)| {
                    o.clone_from(&(a_values[i] * a_values[i]));
                });
            }
        } else {
            // check if inputs are locked
            let (mut shape1, mut shape2) = (
                inputs[0].get_shape().clone().to_vec(),
                inputs[1].get_shape().clone().to_vec(),
            );
            let (mut stride1, mut stride2) = (
                inputs[0].get_stride().clone().to_vec(),
                inputs[1].get_stride().clone().to_vec(),
            );
            Tensor::<T>::broadcast_shapes_linear(
                &mut shape1,
                &mut shape2,
                &mut stride1,
                &mut stride2,
            );

            result_shape = a.get_shape().clone();
            result_values = vec![T::from(0).unwrap(); result_shape.iter().product()];
            let a_lock = a.lock_ref();
            let b_lock = b.lock_ref();
            let a_values = a_lock.get_values_slice();
            let b_values = b_lock.get_values_slice();

            if result_values.len() > 1024 {
                result_values.par_iter_mut().enumerate().for_each(|(i, o)| {
                    let a_idx = Tensor::<T>::get_flat_index(i, &shape1, &stride1);
                    let b_idx = Tensor::<T>::get_flat_index(i, &shape2, &stride2);
                    o.clone_from(&(a_values[a_idx] * b_values[b_idx]));
                });
            } else {
                result_values.iter_mut().enumerate().for_each(|(i, o)| {
                    let a_idx = Tensor::<T>::get_flat_index(i, &shape1, &stride1);
                    let b_idx = Tensor::<T>::get_flat_index(i, &shape2, &stride2);
                    o.clone_from(&(a_values[a_idx] * b_values[b_idx]));
                });
            }
        };
        let result_tensor = Tensor::new(result_values, result_shape);
        let add = Mul {
            input_tensors: Box::new([a, b]),
        };
        result_tensor
            .lock()
            .unwrap()
            .set_op(Arc::new(Mutex::new(add)));
        result_tensor
    }

    fn forward_cuda(_inputs: Box<[SharedTensor<T>]>) -> SharedTensor<T> {
        todo!("CUDA not implemented for Mul operation");
    }

    fn backward(&mut self, upstream: SharedTensor<T>) {
        // Backward implementation for Add operation
        // This is a placeholder and should be implemented as needed
        //
        if Arc::ptr_eq(&self.input_tensors[0], &self.input_tensors[1]) {
            // If both inputs are the same tensor, gradient is 2 * upstream gradient
            let op_grad = {
                let a_shape = self.input_tensors[0].lock().unwrap().get_shape().clone();
                // let a_shape = a.get_shape();
                Mul::forward((
                    Mul::forward((
                        upstream.clone(),
                        Tensor::new(vec![T::from(2).unwrap(); a_shape.iter().product()], a_shape),
                    )),
                    self.input_tensors[0].clone(),
                ))
            };
            Self::apply_grad(self.input_tensors[0].clone(), op_grad.clone());
            Self::recurse_backward(self.input_tensors[0].clone());
        } else {
            let shape_0 = self.input_tensors[0].lock().unwrap().get_shape().clone();
            let shape_1 = self.input_tensors[1].lock().unwrap().get_shape().clone();
            let op_grad_0 = Mul::forward((
                Tensor::new(upstream.lock().unwrap().get_values().clone(), shape_1),
                self.input_tensors[1].clone(),
            ));
            let op_grad_1 = Mul::forward((
                Tensor::new(upstream.lock().unwrap().get_values().clone(), shape_0),
                self.input_tensors[0].clone(),
            ));
            Self::apply_grad(self.input_tensors[0].clone(), op_grad_0.clone());
            Self::apply_grad(self.input_tensors[1].clone(), op_grad_1.clone());
            Self::recurse_backward(self.input_tensors[0].clone());
            Self::recurse_backward(self.input_tensors[1].clone());
        };
    }

    fn get_branches(&self) -> Box<[SharedTensor<T>]> {
        self.input_tensors.clone()
    }
}
