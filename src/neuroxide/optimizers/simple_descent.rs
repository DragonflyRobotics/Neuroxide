use std::collections::HashMap;

use crate::types::{tensor::Tensor, T::TensorElement};

pub struct SimpleDescent<T> {
    lr: Tensor<T>,
    parameters: Vec<Tensor<T>>,
}

impl <T> SimpleDescent<T> 
where
    T: TensorElement
{
    pub fn new(lr: Tensor<T>) -> Self {
        assert!(lr.shape == vec![1], "Learning rate must be a scalar");
        SimpleDescent {
            lr,
            parameters: Vec::new(),
        }
    }

    pub fn add_parameters(&mut self, params: &HashMap<String, Tensor<T>>) {
        for param in params.values() {
            self.parameters.push(param.clone());
        }
    }

    pub fn step(&mut self, grad: &HashMap<i32, Tensor<T>>) {
        for param in &mut self.parameters {
            let grad_param = grad.get(&param.id).unwrap().clone();
            let old_id = param.id;
            *param = param.clone() - grad_param * self.lr.clone();
            param.id = old_id;
            param.clear_graph();
        }
    }
}
