#![allow(warnings)]
use std::sync::{Arc, RwLock};

use neuroxide::ops::add::AddOp;
use neuroxide::ops::mul::MulOp;
use neuroxide::ops::op_generic::Operation;
use neuroxide::types::tensor::Tensor;
use neuroxide::types::device::Device;
use neuroxide::types::tensordb::{DTypes, TensorDB};
use petgraph::dot::Dot;

fn main() {
    let mut db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x = Tensor::new(db.clone(), vec![5.0], vec![1], Device::CPU, true);
    let c1c = Tensor::new(db.clone(), vec![15.0], vec![1], Device::CPU, false);
    let c2c = Tensor::new(db.clone(), vec![6.0], vec![1], Device::CPU, false);
    let r1 = MulOp::forward(&MulOp, &vec![&x, &c1c]);
    let r2 = MulOp::forward(&MulOp, &vec![&x, &c2c]);
    let mut result = AddOp::forward(&AddOp, &vec![&r1, &r2]);
    result = MulOp::forward(&MulOp, &vec![&result, &x]);
    println!("{}", result);

    let grad = result.backward(None);
    for g in grad.keys() {
        println!("{}", grad.get(g).unwrap().data[0]);
    }


}
