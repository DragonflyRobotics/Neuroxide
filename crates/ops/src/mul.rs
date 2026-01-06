use std::sync::{Arc, Mutex};

use crate::op::Operation;
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

        let result_values = if Arc::ptr_eq(&inputs[0], &inputs[1]) {
            result_shape = a.get_shape().clone();
            a.values()
                .iter()
                .zip(a.values().iter())
                .map(|(x, y)| {
                    // Assuming T implements the Add trait
                    *x * *y
                })
                .collect::<Vec<T>>()
        } else {
            (a, b) = Tensor::broadcast_linear(&inputs[0], &inputs[1]);
            result_shape = a.get_shape().clone();
            a.values()
                .iter()
                .zip(b.values().iter())
                .map(|(x, y)| {
                    // Assuming T implements the Add trait
                    *x * *y
                })
                .collect()
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
