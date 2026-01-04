use crate::{
    input::ToTensorInputs,
    op_stub::OperationStub,
    tensor::Tensor,
    tensor_element::{SharedTensor, TensorElement, TensorHandleExt},
};

#[derive(Debug)]
pub struct Unsqueeze<T> {
    pub input_tensors: Box<[SharedTensor<T>]>,
    pub axis: usize,
}

impl<T: TensorElement> OperationStub<T> for Unsqueeze<T> {
    fn forward<I: ToTensorInputs<T>>(inputs: I) -> SharedTensor<T> {
        todo!("Not allowed operation");
    }

    fn backward(&mut self, upstream: SharedTensor<T>) {
        // Backward implementation for Add operation
        let upstream_shape = upstream.get_shape().to_vec();
        let mut target_shape = upstream_shape.clone();
        target_shape.remove(self.axis);
        let grad = Tensor::new(upstream.values().clone(), target_shape);
        self.input_tensors[0].lock().unwrap().set_gradient(grad);
        Self::recurse_backward(self.input_tensors[0].clone());
    }

    fn get_branches(&self) -> Box<[SharedTensor<T>]> {
        self.input_tensors.clone()
    }
}
