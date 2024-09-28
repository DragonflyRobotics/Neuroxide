use crate::types::tensor::Tensor;

pub trait Operation<T: std::marker::Copy>: std::fmt::Debug {
    fn forward(input: &Vec<&Tensor<T>>) -> Tensor<T>;
    fn backward(input: &Vec<&Tensor<T>>, grad: Option<&Tensor<T>>) -> Tensor<T>;
    fn is_binary() -> bool;
}

#[derive(Debug, Clone)]
pub enum Ops {
    TensorEnum,
    AddEnum,
    MulEnum
}
