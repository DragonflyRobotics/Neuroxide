use std::sync::{Arc, RwLock};

use neuroxide::{ops::{div::DivOp, op_generic::Operation}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};


fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let a = Tensor::<f32>::new(&db, vec![0.0, 2.0, 3.0], vec![3], Device::CPU, true);
    let b = Tensor::<f32>::new(&db, vec![4.0, 5.0, 6.0], vec![3], Device::CPU, true);

    let result = DivOp::forward(&vec![&a, &b]);
    println!("{}", result);

    let grad = result.backward(None);
    let da = grad[&a.id].clone();
    let db = grad[&b.id].clone();
    println!("{}", da);
    println!("{}", db);
}
