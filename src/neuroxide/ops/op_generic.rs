use crate::types::tensor::Tensor;

pub trait Operation<T>: std::fmt::Debug {
    fn forward(&self, input: &Vec<&Tensor<T>>) -> Tensor<T>;
    fn clone_box(&self) -> Box<dyn Operation<T>>;
}

impl<T> Clone for Box<dyn Operation<T>> {
    fn clone(&self) -> Box<dyn Operation<T>> {
        self.clone_box()
    }
}
