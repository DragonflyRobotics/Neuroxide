use std::sync::{Arc, RwLock};

use neuroxide::{ops::matmul::MatMulOp, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use neuroxide::ops::op_generic::Operation;

#[macro_use]
extern crate neuroxide;

fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::new(&db, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![3, 2], Device::CPU, true);
    let y = MatMulOp::forward(&vec![&x, &x2]);
    
    println!("{}", x);
    println!("{}", x2);
}
