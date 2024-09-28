use std::collections::HashMap;
use crate::{types::tensor::Tensor, utils::types::print_type_of};

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


pub fn assert_types<T>(dtype: DTypes, data_sample: T) {
    match dtype {
        DTypes::I8 => {
            assert!(print_type_of(&data_sample) == "i8");
        },
        DTypes::I16 => {
            assert!(print_type_of(&data_sample) == "i16");
        },
        DTypes::I32 => {
            assert!(print_type_of(&data_sample) == "i32");
        },
        DTypes::I64 => {
            assert!(print_type_of(&data_sample) == "i64");
        },
        DTypes::I128 => {
            assert!(print_type_of(&data_sample) == "i128");
        },
        DTypes::U8 => {
            assert!(print_type_of(&data_sample) == "u8");
        },
        DTypes::U16 => {
            assert!(print_type_of(&data_sample) == "u16");
        },
        DTypes::U32 => {
            assert!(print_type_of(&data_sample) == "u32");
        },
        DTypes::U64 => {
            assert!(print_type_of(&data_sample) == "u64");
        },
        DTypes::U128 => {
            assert!(print_type_of(&data_sample) == "u128");
        },
        DTypes::F32 => {
            assert!(print_type_of(&data_sample) == "f32");
        },
        DTypes::F64 => {
            assert!(print_type_of(&data_sample) == "f64");
        },
        DTypes::Bool => {
            assert!(print_type_of(&data_sample) == "bool");
        },
        DTypes::Char => {
            assert!(print_type_of(&data_sample) == "char");
        }
    }
}
