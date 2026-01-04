use crate::tensor_element::{SharedTensor, TensorElement};

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
            .cloned()
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

impl ToShapeInputs for &[usize] {
    fn into_shape(self) -> Box<[usize]> {
        self.to_vec().into_boxed_slice()
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
