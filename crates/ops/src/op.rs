use types::{
    device::Device,
    input::{self, ToTensorInputs},
    op_stub::OperationStub,
    tensor_element::{SharedTensor, TensorElement},
};

use crate::add::Add;

pub trait Operation<T: TensorElement>: OperationStub<T> {
    fn apply_grad(tensor: SharedTensor<T>, grad: SharedTensor<T>)
    where
        Self: Sized,
    {
        // No parameters to update in Add operation
        let mut a = tensor.lock().unwrap();
        if let Some(existing_gradient) = a.get_gradient() {
            a.set_gradient(Add::forward((grad.clone(), existing_gradient)));
        } else {
            a.set_gradient(grad.clone());
        }
    }
    fn forward<I: ToTensorInputs<T>>(inputs: I) -> SharedTensor<T>
    where
        Self: Sized,
    {
        let valid_inputs = match Self::check_forward_inputs(inputs) {
            Ok(i) => i,
            Err(err) => panic!("Invalid inputs for operation: {}", err),
        };
        for input in &valid_inputs[1..] {
            assert!(
                valid_inputs[0].lock().unwrap().get_device() == input.lock().unwrap().get_device(),
                "All input tensors must be on the same device"
            );
        }
        let device = valid_inputs[0].lock().unwrap().get_device();
        match device {
            Device::CPU => Self::forward_cpu(valid_inputs),
            Device::CUDA => Self::forward_cuda(valid_inputs.clone()),
        }
    }
}

impl<T: TensorElement, O: OperationStub<T> + ?Sized> Operation<T> for O {}
