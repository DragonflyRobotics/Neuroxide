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
        let input_numel = self.input_tensors[0].get_shape().iter().product();
        let mut grad_values = vec![T::from(0).unwrap(); input_numel];
        {
            let upstream_lock = upstream.lock_ref();
            let input0_lock = self.input_tensors[0].lock_ref();

            let upstream_values = upstream_lock.get_values();
            let upstream_shape = upstream_lock.get_shape();
            let input0_stride = input0_lock.get_stride();
            for output_idx in 0..upstream_shape.iter().product() {
                let output_multi = Tensor::<T>::linear_to_multi(output_idx, &upstream_shape);
                let mut in_multi = vec![0; output_multi.len()];
                for i in 0..output_multi.len() {
                    let ps = &self.parsed_slices[i];
                    in_multi[i] = ps.start + output_multi[i] * ps.step;
                }
                let input_linear: usize = in_multi
                    .iter()
                    .enumerate()
                    .map(|(i, idx)| idx * input0_stride[i])
                    .sum();
                grad_values[input_linear] = grad_values[input_linear] + upstream_values[output_idx];
            }
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
