use std::sync::{Arc, Mutex};

use types::input::ToTensorInputs;
use types::op_stub::OperationStub;
use types::tensor::Tensor;
use types::tensor_element::{SharedTensor, TensorElement};

use crate::mul::Mul;
use crate::op::Operation;

#[derive(Debug)]
pub struct Add<T> {
    input_tensors: Box<[SharedTensor<T>]>,
}

impl<T: TensorElement> OperationStub<T> for Add<T> {
    fn forward<I: ToTensorInputs<T>>(inputs: I) -> SharedTensor<T> {
        let inputs = inputs.into_inputs();
        if inputs.len() != 2 {
            panic!("Add operation requires exactly two input tensors.");
        }

        let a = inputs[0].lock().unwrap();
        let result_values = if Arc::ptr_eq(&inputs[0], &inputs[1]) {
            a.get_values()
                .iter()
                .zip(a.get_values().iter())
                .map(|(x, y)| {
                    // Assuming T implements the Add trait
                    *x + *y
                })
                .collect::<Vec<T>>()
        } else {
            let b = inputs[1].lock().unwrap();
            a.get_values()
                .iter()
                .zip(b.get_values().iter())
                .map(|(x, y)| {
                    // Assuming T implements the Add trait
                    *x + *y
                })
                .collect()
        };
        let result_shape = a.get_shape().clone();
        let result_tensor = Tensor::new(result_values, result_shape);
        let add = Add {
            input_tensors: inputs.clone(),
        };
        result_tensor
            .lock()
            .unwrap()
            .set_op(Arc::new(Mutex::new(add)));
        result_tensor
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
