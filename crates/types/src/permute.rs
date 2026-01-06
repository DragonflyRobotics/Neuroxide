use crate::{
    input::ToTensorInputs,
    op_stub::OperationStub,
    tensor::Tensor,
    tensor_element::{SharedTensor, TensorElement},
};

#[derive(Debug)]
pub struct Permute<T> {
    pub input_tensors: Box<[SharedTensor<T>]>,
    pub reordered_axis: Box<[usize]>,
}

impl<T: TensorElement> OperationStub<T> for Permute<T> {
    fn check_forward_inputs<I: ToTensorInputs<T>>(
        inputs: I,
    ) -> Result<Box<[SharedTensor<T>]>, String>
    where
        Self: Sized,
    {
        Ok(inputs.into_inputs())
    }
    fn forward_cpu(_inputs: Box<[SharedTensor<T>]>) -> SharedTensor<T> {
        todo!("Not allowed operation");
    }

    fn forward_cuda(_inputs: Box<[SharedTensor<T>]>) -> SharedTensor<T> {
        todo!("Not allowed operation");
    }

    fn backward(&mut self, upstream: SharedTensor<T>) {
        // Backward implementation for Add operation
        let inverse_perm = self.reordered_axis.iter().enumerate().fold(
            vec![0; self.reordered_axis.len()],
            |mut acc, (i, &p)| {
                acc[p] = i;
                acc
            },
        );
        let grad = Tensor::permute(&upstream, inverse_perm.into_boxed_slice());
        self.input_tensors[0].lock().unwrap().set_gradient(grad);
        Self::recurse_backward(self.input_tensors[0].clone());
    }

    fn get_branches(&self) -> Box<[SharedTensor<T>]> {
        self.input_tensors.clone()
    }
}
