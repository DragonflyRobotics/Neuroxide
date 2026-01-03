use types::{
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
}

impl<T: TensorElement, O: OperationStub<T> + ?Sized> Operation<T> for O {}
