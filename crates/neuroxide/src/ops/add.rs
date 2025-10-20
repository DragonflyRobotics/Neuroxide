use std::sync::{Arc, Mutex};

use crate::{
    ops::op::Operation,
    types::{tensor::Tensor, tensor_element::TensorElement},
};

#[derive(Debug)]
pub struct Add<T> {
    input_tensors: Box<[Arc<Mutex<Tensor<T>>>]>,
}

impl<T: TensorElement + 'static> Operation<T> for Add<T> {
    fn forward(inputs: Box<[Arc<Mutex<Tensor<T>>>]>) -> Arc<Mutex<Tensor<T>>> {
        let result_data = {
            let a_values = {
                let a = inputs[0].lock().unwrap();
                a.get_values().clone() // make an owned copy
            }; // lock released here
            let b_values = {
                let b = inputs[1].lock().unwrap();
                b.get_values().clone() // make an owned copy
            }; // lock released here

            let result_values: Vec<T> = a_values
                .iter()
                .zip(b_values.iter())
                .map(|(x, y)| {
                    // Assuming T implements the Add trait
                    *x + *y
                })
                .collect();

            result_values
        };
        let result_shape = {
            let a = inputs[0].lock().unwrap();
            a.get_shape().clone()
        };
        let result_tensor = Tensor::new(result_data, result_shape);
        let add = Add {
            input_tensors: inputs.clone(),
        };
        result_tensor
            .lock()
            .unwrap()
            .set_op(Arc::new(Mutex::new(add)));
        result_tensor
    }

    fn backward(&mut self, upstream: Arc<Mutex<Tensor<T>>>) {
        // Backward implementation for Add operation
        // This is a placeholder and should be implemented as needed
        //
        if Arc::ptr_eq(&self.input_tensors[0], &self.input_tensors[1]) {
            // If both inputs are the same tensor, gradient is 2 * upstream gradient
            println!("Fix me later: {:?}", upstream);
            let op_grad = {
                let a = self.input_tensors[0].lock().unwrap();
                Tensor::new(
                    vec![T::from(2).unwrap(); a.get_values().len()],
                    a.get_shape().clone(),
                )
            };
            Self::apply_grad(self.input_tensors[0].clone(), op_grad.clone());
            Self::recurse_backward(self.input_tensors[0].clone());
        } else {
            let op_grad;
            {
                op_grad = Tensor::new(
                    upstream.lock().unwrap().get_values().clone(),
                    self.input_tensors[0].lock().unwrap().get_shape().clone(),
                );
                Self::apply_grad(self.input_tensors[0].clone(), op_grad.clone());
                Self::apply_grad(self.input_tensors[1].clone(), op_grad.clone());
            }
            Self::recurse_backward(self.input_tensors[0].clone());
            Self::recurse_backward(self.input_tensors[1].clone());
        };
    }

    fn get_branches(&self) -> Box<[Arc<Mutex<Tensor<T>>>]> {
        self.input_tensors.clone()
    }
}

impl<T: TensorElement + 'static> Add<T> {
    fn apply_grad(tensor: Arc<Mutex<Tensor<T>>>, grad: Arc<Mutex<Tensor<T>>>) {
        // No parameters to update in Add operation
        let mut a = tensor.lock().unwrap();
        if let Some(existing_gradient) = a.get_gradient() {
            a.set_gradient(Add::forward(Box::new([grad.clone(), existing_gradient])));
        } else {
            a.set_gradient(grad.clone());
        }
    }

    fn recurse_backward(tensor: Arc<Mutex<Tensor<T>>>) {
        let binding = tensor.lock().unwrap();
        if let Some(op_arc) = binding.get_op() {
            let mut op = op_arc.lock().unwrap();
            if let Some(grad) = binding.get_gradient() {
                op.backward(grad);
            }
        }
    }
}
