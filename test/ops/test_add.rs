use std::sync::{Arc, RwLock};

use neuroxide::{ops::{add::AddOp, mul::MulOp, op_generic::Operation as _}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use approx::relative_eq;

#[test]
fn forward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let mut c1c = Tensor::new(db.clone(), vec![15.0], vec![1], Device::CPU, false);
    let mut c2c = Tensor::new(db.clone(), vec![6.0], vec![1], Device::CPU, false);
    let mut result = AddOp::forward(&AddOp, &vec![&c1c, &c2c]);
    assert_eq!(result.data[0], 21.0);

    c1c = Tensor::new(db.clone(), vec![15.0, 4.1, 2.3, 34.1, 12.2], vec![2,2], Device::CPU, false); 
    c2c = Tensor::new(db.clone(), vec![6.0, 3.1, 1.3, 4.1, 2.2], vec![2,2], Device::CPU, false);
    result = AddOp::forward(&AddOp, &vec![&c1c, &c2c]);
    for i in 0..result.data.len() {
        assert!(relative_eq!(result.data[i], c1c.data[i] + c2c.data[i], epsilon = f64::EPSILON));
    }
}

#[test]
fn backward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x = Tensor::new(db.clone(), vec![5.0], vec![1], Device::CPU, true);
    let c1c = Tensor::new(db.clone(), vec![15.0], vec![1], Device::CPU, false);
    let c2c = Tensor::new(db.clone(), vec![6.0], vec![1], Device::CPU, false);
    let r1 = MulOp::forward(&MulOp, &vec![&x, &c1c]);
    let r2 = MulOp::forward(&MulOp, &vec![&x, &c2c]);
    let mut result = AddOp::forward(&AddOp, &vec![&r1, &r2]);
    result = MulOp::forward(&MulOp, &vec![&result, &x]);
    assert!(relative_eq!(result.data[0], 525.0));

    let grad = result.backward(None);
    assert!(relative_eq!(grad.get(&x.id).unwrap().data[0], 210.0));
}
