use crate::{
    input::ToTensorInputs,
    op_stub::OperationStub,
    tensor::Tensor,
    tensor_element::{SharedTensor, TensorElement, TensorHandleExt},
};

#[derive(Debug)]
pub struct Broadcast<T> {
    pub input_tensors: Box<[SharedTensor<T>]>,
    pub arg_params: BroadcastParams,
}

#[derive(Debug, Clone)]
pub struct BroadcastParams {
    pub axes_to_sum: Vec<bool>,
    pub axes_to_remove: Vec<bool>,
}

impl<T: TensorElement> OperationStub<T> for Broadcast<T> {
    fn forward<I: ToTensorInputs<T>>(inputs: I) -> SharedTensor<T> {
        todo!("Not allowed operation");
    }

    fn backward(&mut self, upstream: SharedTensor<T>) {
        let mut grad = upstream.clone();

        let mut sum_axes = Vec::new();
        let mut squeeze_axes = Vec::new();

        for i in 0..self.arg_params.axes_to_sum.len() {
            if self.arg_params.axes_to_sum[i] {
                sum_axes.push(i);
            }
            if self.arg_params.axes_to_remove[i] {
                squeeze_axes.push(i);
            }
        }

        // CRITICAL
        sum_axes.sort_unstable_by(|a, b| b.cmp(a));
        squeeze_axes.sort_unstable_by(|a, b| b.cmp(a));

        for axis in sum_axes {
            grad = Tensor::sum(&grad, axis); // keepdim = true
        }

        for axis in squeeze_axes {
            grad = Tensor::squeeze(&grad, axis);
        }

        self.input_tensors[0].lock().unwrap().set_gradient(grad);

        Self::recurse_backward(self.input_tensors[0].clone());
    }

    fn get_branches(&self) -> Box<[SharedTensor<T>]> {
        self.input_tensors.clone()
    }
}
