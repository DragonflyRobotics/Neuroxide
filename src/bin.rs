// use neuroxide::idk::cuda_main;
// struct test {
//     a: *mut f32
// }

use std::sync::{Arc, RwLock};
use std::time::{SystemTime, UNIX_EPOCH};
use neuroxide::ops::add::AddOp;
use neuroxide::ops::cos::CosOp;
use neuroxide::ops::div::DivOp;
use neuroxide::ops::mul::MulOp;
use neuroxide::ops::op_generic::Operation;
use neuroxide::ops::pow::PowOp;
use neuroxide::ops::sin::SinOp;
use neuroxide::ops::sub::SubOp;
use neuroxide::{ops::ln::LnOp, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};


fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let mut c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
    let mut c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
    let mut result = DivOp::forward(&vec![&c1c, &c2c]);
    println!("result: {}", result);

    c1c = Tensor::new(&db, vec![15.0, 4.1, 2.3, 34.1], vec![2,2], Device::CPU, false); 
    c2c = Tensor::new(&db, vec![6.0, 3.1, 1.3, 4.1], vec![2,2], Device::CPU, false);
    result = DivOp::forward(&vec![&c1c, &c2c]);
    println!("result: {}", result);
}


// fn main() {
//     // cuda_main();
//     let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
//     let db = Arc::new(RwLock::new(TensorDB::<f32>::new(DTypes::F32)));
//     let a = Tensor::new(&db, vec![3.0], vec![1], Device::CUDA, true);
//     let b = Tensor::new(&db, vec![6.0], vec![1], Device::CUDA, false);
//     let mut c = AddOp::forward(&vec![&a, &b]);
//     let mut d = SubOp::forward(&vec![&a, &c]);
//     let e = MulOp::forward(&vec![&a, &d]);
//     let f = DivOp::forward(&vec![&a, &e]);
//     let g = PowOp::forward(&vec![&a, &f]);
//
//     let mut h = SinOp::forward(&vec![&g]);
//
//     let mut grad = h.backward(None);
//     for g in grad.values_mut() {
//         println!("grad fin: {}", g);
//     }
//     
//     let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
//     // tensor_cpu(&db);
//     // c.cpu();
//     // println!("a: {}", a);
//     // println!("b: {}", b);
//     // println!("c: {}", c);
//     // println!("d: {}", d);
//     // println!("e: {}", e);
//     // println!("f: {}", f);
//     // println!("g: {}", g);
//     //
//     //
//     // println!("h: {}", h);
//     println!("Time: {:?}", end - start);
// }
