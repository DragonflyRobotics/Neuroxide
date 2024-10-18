use std::sync::{Arc, RwLock};

use neuroxide::{ops::{cos::CosOp, op_generic::Operation as _, sin::SinOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use approx::relative_eq;

#[test]
fn forward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::<f64>::new(&db, vec![0.0, 3.14/6.0, 3.14/4.0, 3.14/3.0, 3.14], vec![1], Device::CPU, false);
    let result = CosOp::forward(&vec![&x]);

    for i in 0..result.data.len() {
        assert!(relative_eq!(result.data[i], x.data[i].cos(), epsilon = f64::EPSILON));
    }

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x = Tensor::<f32>::new(&db, vec![0.0, 3.14/6.0, 3.14/4.0, 3.14/3.0, 3.14], vec![1], Device::CPU, false);
    let result = CosOp::forward(&vec![&x]);

    for i in 0..result.data.len() {
        assert!(relative_eq!(result.data[i], x.data[i].cos(), epsilon = f32::EPSILON));
    }
}

#[test]
fn backward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::<f64>::new(&db, vec![0.0, 3.14/6.0, 3.14/4.0, 3.14/3.0, 3.14], vec![1], Device::CPU, true);
    let result = CosOp::forward(&vec![&x]);
    let grad = result.backward(None);

    for i in 0..result.data.len() {
        assert!(relative_eq!(grad.get(&x.id).unwrap().data[i], -x.data[i].sin(), epsilon = f64::EPSILON));
    }

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x = Tensor::<f32>::new(&db, vec![0.0, 3.14/6.0, 3.14/4.0, 3.14/3.0, 3.14], vec![1], Device::CPU, true);
    let result = CosOp::forward(&vec![&x]);
    let grad = result.backward(None);

    for i in 0..result.data.len() {
        assert!(relative_eq!(grad.get(&x.id).unwrap().data[i], -x.data[i].sin(), epsilon = f32::EPSILON));
    }
}


#[test]
#[should_panic]
fn downcasts_i8() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I8)));
    let x = Tensor::<i8>::new(&db, vec![0, 1, 2, 3, 4], vec![1], Device::CPU, false);
    let _ = CosOp::forward(&vec![&x]);
}

#[test]
#[should_panic]
fn downcasts_i16() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I16)));
    let x = Tensor::<i16>::new(&db, vec![0, 1, 2, 3, 4], vec![1], Device::CPU, false);
    let _ = CosOp::forward(&vec![&x]);
}

#[test]
#[should_panic]
fn downcasts_i32() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::<i32>::new(&db, vec![0, 1, 2, 3, 4], vec![1], Device::CPU, false);
    let _ = CosOp::forward(&vec![&x]);
}

#[test]
#[should_panic]
fn downcasts_i64() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I64)));
    let x = Tensor::<i64>::new(&db, vec![0, 1, 2, 3, 4], vec![1], Device::CPU, false);
    let _ = CosOp::forward(&vec![&x]);
}

#[test]
#[should_panic]
fn downcasts_i128() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I128)));
    let x = Tensor::<i128>::new(&db, vec![0, 1, 2, 3, 4], vec![1], Device::CPU, false);
    let _ = CosOp::forward(&vec![&x]);
}

#[test]
#[should_panic]
fn downcasts_u8() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::U8)));
    let x = Tensor::<u8>::new(&db, vec![0, 1, 2, 3, 4], vec![1], Device::CPU, false);
    let _ = CosOp::forward(&vec![&x]);
}

#[test]
#[should_panic]
fn downcasts_u16() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::U16)));
    let x = Tensor::<u16>::new(&db, vec![0, 1, 2, 3, 4], vec![1], Device::CPU, false);
    let _ = CosOp::forward(&vec![&x]);
}

#[test]
#[should_panic]
fn downcasts_u32() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::U32)));
    let x = Tensor::<u32>::new(&db, vec![0, 1, 2, 3, 4], vec![1], Device::CPU, false);
    let _ = CosOp::forward(&vec![&x]);
}

#[test]
#[should_panic]
fn downcasts_u64() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::U64)));
    let x = Tensor::<u64>::new(&db, vec![0, 1, 2, 3, 4], vec![1], Device::CPU, false);
    let _ = CosOp::forward(&vec![&x]);
}

#[test]
#[should_panic]
fn downcasts_u128() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::U128)));
    let x = Tensor::<u128>::new(&db, vec![0, 1, 2, 3, 4], vec![1], Device::CPU, false);
    let _ = CosOp::forward(&vec![&x]);
}

