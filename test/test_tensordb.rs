use std::sync::{Arc, RwLock};

use neuroxide::{ops::op_generic::Ops, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};

#[test]
fn new() {
    let db = Arc::new(RwLock::new(TensorDB::<f32>::new(DTypes::F32)));
    assert!(db.read().unwrap().get_dtype() == DTypes::F32);
}

#[test]
fn insert_get() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x = Tensor {
        id: 0,
        data: vec![5.0],
        shape: vec![1],
        device: Device::CPU,
        requires_grad: true,
        op_chain: Default::default(),
        op_head: 0,
        op: Ops::TensorEnum,
        dtype: db.clone()
    };
    db.write().unwrap().insert(x.clone());
    assert!(db.read().unwrap().get(0).unwrap().data == vec![5.0]);
}

#[test]
fn get_dtype() {
    let db = Arc::new(RwLock::new(TensorDB::<f32>::new(DTypes::F32)));
    assert!(db.read().unwrap().get_dtype() == DTypes::F32);
}
