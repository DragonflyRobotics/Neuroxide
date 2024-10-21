use std::sync::{Arc, RwLock};

use neuroxide::{ops::{cos::CosOp, div::DivOp, mul::MulOp, op_generic::Operation, pow::PowOp, sin::SinOp, sub::SubOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use petgraph::dot::Dot;


fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    let c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
    let c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
    
    let result = x.clone() * (SinOp::forward(&vec![&(c2c.clone()*x.clone())]) - c1c*x.clone()) - CosOp::forward(&vec![&PowOp::forward(&vec![&x.clone(), &c2c.clone()])]) + PowOp::forward(&vec![&c2c.clone(), &x.clone()]); 
    println!("{}", result);
    let grad = result.backward(None);
    println!("{}", grad.get(&x.id).unwrap());

}
