use std::sync::{Arc, RwLock};

use neuroxide::{ops::matmul::MatMulOp, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use neuroxide::ops::op_generic::Operation;

#[macro_use]
extern crate neuroxide;


fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![3, 2], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    println!("{}", c);
    println!("====================");
    let grad = c.backward(None);
    println!("{}", grad.get(&x.id).unwrap());
    println!("====================");
    
}
