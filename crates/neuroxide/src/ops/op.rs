use crate::{
    ops::add::Add,
    types::tensor_element::{SharedTensor, TensorElement},
};

pub trait Operation<T: TensorElement> {
    fn forward<I: ToTensorInputs<T>>(inputs: I) -> SharedTensor<T>
    where
        Self: Sized;
    fn backward(&mut self, upstream: SharedTensor<T>);
    fn get_branches(&self) -> Box<[SharedTensor<T>]>;
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

pub trait ToTensorInputs<T: TensorElement> {
    fn into_inputs(self) -> Box<[SharedTensor<T>]>;
}

impl<T: TensorElement> ToTensorInputs<T> for SharedTensor<T> {
    fn into_inputs(self) -> Box<[SharedTensor<T>]> {
        Box::new([self])
    }
}

impl<T: TensorElement> ToTensorInputs<T> for &SharedTensor<T> {
    fn into_inputs(self) -> Box<[SharedTensor<T>]> {
        Box::new([self.clone()])
    }
}

impl<T: TensorElement> ToTensorInputs<T> for (SharedTensor<T>, SharedTensor<T>) {
    fn into_inputs(self) -> Box<[SharedTensor<T>]> {
        Box::new([self.0, self.1])
    }
}

impl<T: TensorElement> ToTensorInputs<T> for (&SharedTensor<T>, &SharedTensor<T>) {
    fn into_inputs(self) -> Box<[SharedTensor<T>]> {
        Box::new([self.0.clone(), self.1.clone()])
    }
}

impl<T: TensorElement> ToTensorInputs<T> for Vec<SharedTensor<T>> {
    fn into_inputs(self) -> Box<[SharedTensor<T>]> {
        self.into_boxed_slice()
    }
}

impl<T: TensorElement> ToTensorInputs<T> for Vec<&SharedTensor<T>> {
    fn into_inputs(self) -> Box<[SharedTensor<T>]> {
        self.into_iter()
            .map(|t| t.clone())
            .collect::<Vec<_>>()
            .into_boxed_slice()
    }
}

pub trait ToShapeInputs {
    fn into_shape(self) -> Box<[usize]>;
}

impl ToShapeInputs for Vec<usize> {
    fn into_shape(self) -> Box<[usize]> {
        self.into_boxed_slice()
    }
}

impl<const N: usize> ToShapeInputs for [usize; N] {
    fn into_shape(self) -> Box<[usize]> {
        self.to_vec().into_boxed_slice()
    }
}

impl ToShapeInputs for Box<[usize]> {
    fn into_shape(self) -> Box<[usize]> {
        self
    }
}

pub trait ToTensorValueInputs<T> {
    fn into_values(self) -> Vec<T>;
}

impl<T> ToTensorValueInputs<T> for Vec<T> {
    fn into_values(self) -> Vec<T> {
        self
    }
}

impl<T: TensorElement, const N: usize> ToTensorValueInputs<T> for [T; N] {
    fn into_values(self) -> Vec<T> {
        self.to_vec()
    }
}
