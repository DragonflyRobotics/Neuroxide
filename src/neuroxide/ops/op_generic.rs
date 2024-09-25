use crate::types::tensor::Tensor;

pub trait Operation<T: std::marker::Copy>: std::fmt::Debug {
    fn forward(input: &Vec<&Tensor<T>>) -> Tensor<T>;
    fn backward(input: &Vec<&Tensor<T>>, grad: Option<&Tensor<T>>) -> Tensor<T>;
    // fn clone_box() -> Box<dyn Operation<T>>;
}

// impl<T: std::marker::Copy> Clone for Box<dyn Operation<T>> {
//     fn clone(&self) -> Box<dyn Operation<T>> {
//         self.clone_box()
//     }
// }

#[derive(Debug, Clone)]
pub enum Ops {
    TensorEnum,
    AddEnum,
    MulEnum
}
