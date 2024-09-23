use std::collections::HashMap;
use crate::types::tensor::Tensor;

#[derive(Clone, PartialEq)]
pub enum DTypes {
    I8,
    I16,
    I32,
    I64,
    I128,
    U8,
    U16,
    U32,
    U64,
    U128,
    F32,
    F64,
    Bool,
    Char
}


#[derive(Clone)]
pub struct TensorDB<T> {
    tensors: HashMap<i32, Tensor<T>>,
    dtype: DTypes
}


impl <T> TensorDB<T> {
    pub fn new(dtype: DTypes) -> TensorDB<T> {
        TensorDB {
            tensors: HashMap::new(),
            dtype
        }
    }

    pub fn insert(&mut self, tensor: Tensor<T>) {
        self.tensors.insert(tensor.id, tensor);
    }

    pub fn get(&self, id: i32) -> Option<&Tensor<T>> {
        self.tensors.get(&id)
    }

    pub fn get_dtype(&self) -> DTypes {
        self.dtype.clone()
    }
}

