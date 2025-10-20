use std::sync::{Arc, Mutex};

use crate::types::{tensor::Tensor, tensor_element::TensorElement};

pub trait Operation<T: TensorElement> {
    fn forward(inputs: Box<[Arc<Mutex<Tensor<T>>>]>) -> Arc<Mutex<Tensor<T>>>
    where
        Self: Sized;
    fn backward(&mut self, upstream: Arc<Mutex<Tensor<T>>>);
    fn get_branches(&self) -> Box<[Arc<Mutex<Tensor<T>>>]>;
}
