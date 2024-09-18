use std::{collections::HashMap, sync::Mutex};
use crate::types::tensor::Tensor;
use once_cell::sync::Lazy;

#[derive(Debug, Clone)]
pub struct TensorDB<T> {
    tensors: HashMap<i32, Tensor<T>>,
}

pub enum DTypes {
    I32,
    F32,
    F64
}

impl <T> TensorDB<T> {
    pub fn new() -> TensorDB<T> {
        TensorDB {
            tensors: HashMap::new()
        }
    }

    pub fn insert(&mut self, tensor: Tensor<T>) {
        self.tensors.insert(tensor.id, tensor);
    }

    pub fn get(&self, id: i32) -> Option<&Tensor<T>> {
        self.tensors.get(&id)
    }
}
pub fn make_tensor_db(dtype: DTypes) {
    match dtype {
        DTypes::I32 => {
            static TENSOR_DB: Lazy<Mutex<TensorDB<i32>>> = Lazy::new(|| Mutex::new(TensorDB::new()));
        },
        DTypes::F32 => {
            static TENSOR_DB: Lazy<Mutex<TensorDB<f32>>> = Lazy::new(|| Mutex::new(TensorDB::new()));
        },
        DTypes::F64 => {
            static TENSOR_DB: Lazy<Mutex<TensorDB<f64>>> = Lazy::new(|| Mutex::new(TensorDB::new()));
        }
    }
}
