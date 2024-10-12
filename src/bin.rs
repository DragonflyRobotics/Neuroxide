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

use neuroxide::utils::types::print_type_of;

fn main() {
    use std::time::Instant;
    let now = Instant::now();
    let mut db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x = Tensor::<f32>::new(&db, vec![3.14159/4.0], vec![1], Device::CPU, true);

    let result = SinOp::forward(&vec![&x]);
    println!("{}", result);
    let dot = Dot::new(&result.op_chain);
    println!("{:?}", dot);
    let grad = result.backward(None);
    for n in grad.keys() {
        let t = grad.get(n).unwrap();
        println!("Node: {}, Data: {:?}", n, t.data);
        let dot = Dot::new(&t.op_chain);
        println!("{:?}", dot);
    }


    // let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    // let x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    // let c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
    // let c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
    // let r1 = MulOp::forward(&vec![&x, &c1c]);
    // let r2 = MulOp::forward(&vec![&x, &c2c]);
    // let mut result = AddOp::forward(&vec![&r1, &r2]);
    // result = MulOp::forward(&vec![&result, &x]);
    // let dot = Dot::new(&result.op_chain);
    // println!("{:?}", dot);
    //
    //
    // let grad = result.backward(None);
    // for n in grad.keys() {
    //     let t = grad.get(n).unwrap();
    //     println!("Node: {}, Data: {:?}", n, t.data);
    //     let dot = Dot::new(&t.op_chain);
    //     println!("{:?}", dot);
    // }

}
