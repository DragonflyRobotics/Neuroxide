#![allow(warnings)]
use std::sync::{Arc, RwLock};

use neuroxide::ops::add::AddOp;
use neuroxide::ops::mul::MulOp;
use neuroxide::ops::op_generic::Operation;
use neuroxide::types::tensor::Tensor;
use neuroxide::types::device::Device;
use neuroxide::types::tensordb::{DTypes, TensorDB};
use petgraph::dot::Dot;

// fn main() {
//     use std::time::Instant;
//     let now = Instant::now();
//     let mut db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
//     let x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
//     let c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
//     let c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
//     let r1 = MulOp::forward(&vec![&x, &c1c]);
//     let r2 = MulOp::forward(&vec![&x, &c2c]);
//     let mut result = AddOp::forward(&vec![&r1, &r2]);
//     result = MulOp::forward(&vec![&result, &x]);
//     println!("{}", result);
//
//     let grad = result.backward(None);
//     for g in grad.keys() {
//         println!("{}", grad.get(g).unwrap().data[0]);
//     }
//
//     let elapsed = now.elapsed();
//     println!("Elapsed: {:.2?}", elapsed);
//
// }
fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let mut c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
    let mut c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
    let _result = MulOp::forward(&vec![&c1c, &c2c]);

    c1c = Tensor::new(&db, vec![15.0, 4.1, 2.3, 34.1, 12.2], vec![5,1], Device::CPU, false); 
    c2c = Tensor::new(&db, vec![6.0, 3.1, 1.3, 4.1, 2.2], vec![5,1], Device::CPU, false);
    let result = MulOp::forward(&vec![&c1c, &c2c]);
    println!("{}", result);
}
