use std::sync::{Arc, RwLock};

use neuroxide::{ops::{op_generic::Operation as _, sin::SinOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use approx::relative_eq;

#[test]
fn forward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::<f64>::new(&db, vec![0.0, 3.14/6.0, 3.14/4.0, 3.14/3.0, 3.14], vec![1], Device::CPU, false);
    let result = SinOp::forward(&vec![&x]);

    for i in 0..result.data.len() {
        assert!(relative_eq!(result.data[i], x.data[i].sin(), epsilon = f64::EPSILON));
    }
}

#[test]
fn backward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::<f64>::new(&db, vec![0.0, 3.14/6.0, 3.14/4.0, 3.14/3.0, 3.14], vec![1], Device::CPU, true);
    let result = SinOp::forward(&vec![&x]);
    let grad = result.backward(None);

    for i in 0..result.data.len() {
        assert!(relative_eq!(grad.get(&x.id).unwrap().data[i], x.data[i].cos(), epsilon = f64::EPSILON));
    }
}
