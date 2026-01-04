use crate::{
    input::ToTensorInputs,
    op_stub::OperationStub,
    tensor::{ParsedSlice, Tensor},
    tensor_element::{SharedTensor, TensorElement, TensorHandleExt},
};

#[derive(Debug)]
pub struct Slice<T> {
    pub input_tensors: Box<[SharedTensor<T>]>,
    pub parsed_slices: Vec<ParsedSlice>,
}

impl<T: TensorElement> OperationStub<T> for Slice<T> {
    fn forward<I: ToTensorInputs<T>>(inputs: I) -> SharedTensor<T> {
        todo!("Not allowed operation");
    }

    fn backward(&mut self, upstream: SharedTensor<T>) {
        let input_numel = self.input_tensors[0].get_shape().iter().product();
        let mut grad_values = vec![T::from(0).unwrap(); input_numel];
        for output_idx in 0..upstream.get_shape().iter().product() {
            let output_multi = Tensor::<T>::linear_to_multi(output_idx, &upstream.get_shape());
            let mut in_multi = vec![0; output_multi.len()];
            for i in 0..output_multi.len() {
                let ps = &self.parsed_slices[i];
                in_multi[i] = ps.start + output_multi[i] * ps.step;
            }
            let input_linear: usize = in_multi
                .iter()
                .enumerate()
                .map(|(i, idx)| idx * self.input_tensors[0].get_stride()[i])
                .sum();
            grad_values[input_linear] = grad_values[input_linear] + upstream.values()[output_idx];
        }
        let grad_tensor = Tensor::new(grad_values, self.input_tensors[0].get_shape().to_vec());
        self.input_tensors[0]
            .lock()
            .unwrap()
            .set_gradient(grad_tensor);
        Self::recurse_backward(self.input_tensors[0].clone());
    }

    fn get_branches(&self) -> Box<[SharedTensor<T>]> {
        self.input_tensors.clone()
    }
}
