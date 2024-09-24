extern crate test;

use std::sync::{Arc, RwLock};

use neuroxide::{ops::op_generic::Ops, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};

#[bench]
fn new(b: &mut test::Bencher) {
    b.iter(|| {
        let _ = Arc::new(RwLock::new(TensorDB::<f32>::new(DTypes::F32)));
    });
}

#[bench]
fn insert_get(b: &mut test::Bencher) {
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
    b.iter(|| {
        db.write().unwrap().insert(x.clone());
        db.read().unwrap().get(0).unwrap();
    });
}

#[bench]
fn get_dtype(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::<f32>::new(DTypes::F32)));
    b.iter(|| {
        db.read().unwrap().get_dtype();
    });
}
