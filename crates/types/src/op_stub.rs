use crate::input::ToTensorInputs;
use crate::tensor_element::SharedTensor;
use crate::tensor_element::TensorElement;

pub trait OperationStub<T: TensorElement> {
    fn forward<I: ToTensorInputs<T>>(inputs: I) -> SharedTensor<T>
    where
        Self: Sized;
    fn backward(&mut self, upstream: SharedTensor<T>);
    fn get_branches(&self) -> Box<[SharedTensor<T>]>;
    fn recurse_backward(tensor: SharedTensor<T>)
    where
        Self: Sized,
    {
        let binding = tensor.lock().unwrap();
        if let Some(op_arc) = binding.get_op() {
            let mut op = op_arc.lock().unwrap();
            if let Some(grad) = binding.get_gradient() {
                op.backward(grad);
            }
        }
    }
}
