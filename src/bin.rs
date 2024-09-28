#![allow(warnings)]
use std::sync::{Arc, RwLock};

use neuroxide::ops::add::AddOp;
use neuroxide::ops::mul::MulOp;
use neuroxide::ops::sin::SinOp;
use neuroxide::ops::op_generic::Operation;
use neuroxide::types::tensor::Tensor;
use neuroxide::types::device::Device;
use neuroxide::types::tensordb::{DTypes, TensorDB};
use petgraph::dot::Dot;

fn main() {
    use std::time::Instant;
    let now = Instant::now();
    let mut db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x = Tensor::new(&db, vec![3.14159], vec![1], Device::CPU, true);
    let c = Tensor::new(&db, vec![4.0], vec![1], Device::CPU, true);
    let mut result = AddOp::forward(&vec![&x, &c]);

    result = SinOp::forward(&vec![&result]);
    let dot = Dot::new(&result.op_chain);
    println!("{:?}", dot);

}
