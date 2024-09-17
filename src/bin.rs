#![allow(warnings)]
use neuroxide::ops::add::AddOp;
use neuroxide::ops::mul::MulOp;
use neuroxide::ops::op_generic::Operation;
use neuroxide::types::tensor::Tensor;
use neuroxide::types::device::Device;
use neuroxide::types::tensordb::TensorDB;
use petgraph::dot::Dot;

fn main() {
    let mut db = TensorDB::new();
    let a = Tensor::new(&mut db, vec![1.0], vec![1], Device::CPU, true);
    let b = Tensor::new(&mut db, vec![4.0], vec![1], Device::CPU, true);
    let mut result = MulOp.forward(&mut db, &vec![&a, &b]);
    result = MulOp.forward(&mut db, &vec![&result, &a]);
    println!("{:?}", result);
    println!("{:?}", Dot::new(&result.op_chain));
    let grad = result.backward(&mut db, None);
    println!("Result: {:?}", result.clone());
    println!("Grad: {:?}", grad);
}
