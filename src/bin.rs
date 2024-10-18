use std::sync::{Arc, RwLock};

use neuroxide::{ops::{add::AddOp, cos::CosOp, op_generic::Operation}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use petgraph::dot::Dot;


fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x1 = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, true);
    let x3 = Tensor::new(&db, vec![7.0], vec![1], Device::CPU, true);
    let x4 = Tensor::new(&db, vec![8.0], vec![1], Device::CPU, true);

    let result = x1.clone() * (x2.clone() + x3) + x4;
    let dot = Dot::new(&result.op_chain); 
    println!("{}", dot.to_string());
    let grad = result.backward(Some(vec![x2.id.clone()]));
    println!("{}", grad.get(&x1.id).unwrap().data[0]);
}
