use crate::types::{device::Device, t::TensorElement, tensor::Tensor};

pub trait Operation<T: TensorElement>: std::fmt::Debug {
    fn forward(input: &Vec<&Tensor<T>>) -> Tensor<T>;
    fn backward(input: &Vec<&Tensor<T>>, grad: Option<&Tensor<T>>, device: Device) -> Tensor<T>;
}

#[derive(Debug, Clone)]
pub enum Ops {
    TensorEnum,
    AddEnum,
    MulEnum,
    SinEnum,
    CosEnum,
    PowEnum,
    LnEnum,
    DivEnum,
    SubEnum,
    MatMulEnum,
}
