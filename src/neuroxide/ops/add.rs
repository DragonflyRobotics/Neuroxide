use crate::ops::op_generic::Operation;
use crate::types::tensor::Tensor;
use std::ops::Add;

#[derive(Debug, Clone)]
pub struct AddOp;

impl<T> Operation<T> for AddOp
where
    T: Add<Output = T> + Copy + Default,
{
    fn forward(&self, inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);
        inputs[0].clone() + inputs[1].clone()
    }

    fn clone_box(&self) -> Box<dyn Operation<T>> {
        Box::new(AddOp)
    }
}

