use crate::{
    input::ToTensorInputs,
    op_stub::OperationStub,
    tensor::Tensor,
    tensor_element::{SharedTensor, TensorElement, TensorHandleExt},
};

#[derive(Debug)]
pub struct AxisSum<T> {
    pub input_tensors: Box<[SharedTensor<T>]>,
    pub axis: usize,
}

impl<T: TensorElement> OperationStub<T> for AxisSum<T> {
    fn forward<I: ToTensorInputs<T>>(inputs: I) -> SharedTensor<T> {
        todo!("Not allowed operation");
    }

    fn backward(&mut self, upstream: SharedTensor<T>) {
        // Backward implementation for Add operation
        let (c, _) = Tensor::broadcast_linear(&upstream, &self.input_tensors[0]);
        self.input_tensors[0].lock().unwrap().set_gradient(c);
        Self::recurse_backward(self.input_tensors[0].clone());
    }

    fn get_branches(&self) -> Box<[SharedTensor<T>]> {
        self.input_tensors.clone()
    }
}
