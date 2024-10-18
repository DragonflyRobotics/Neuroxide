use std::sync::{Arc, RwLock};

use neuroxide::{ops::{add::AddOp, cos::CosOp, mul::MulOp, op_generic::Operation}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use petgraph::dot::Dot;


fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x1 = Tensor::new(&db, vec![5i32; 10000000], vec![1], Device::CUDA, true);
    let x2 = Tensor::new(&db, vec![6i32; 10000000], vec![1], Device::CUDA, false);

    let result = MulOp::forward(&vec![&x1, &x2]);
    for _ in 0..5 {
        let result = MulOp::forward(&vec![&x1, &x2]); 
    }
    println!("cuda {:?}", result.data[0]);
    let grad = result.backward(None);
    println!("{}", grad.get(&x1.id).unwrap().data[0]);
}

