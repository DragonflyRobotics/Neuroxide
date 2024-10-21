use std::sync::{Arc, RwLock};


use neuroxide::{ops::{mul::MulOp, op_generic::Operation as _, sub::SubOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use approx::relative_eq;

#[test]
fn forward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let mut c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
    let mut c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
    let mut result = SubOp::forward(&vec![&c1c, &c2c]);
    assert_eq!(result.data[0], 9.0);

    c1c = Tensor::new(&db, vec![15.0, 4.1, 2.3, 34.1, 12.2], vec![2,2], Device::CPU, false); 
    c2c = Tensor::new(&db, vec![6.0, 3.1, 1.3, 4.1, 2.2], vec![2,2], Device::CPU, false);
    result = SubOp::forward(&vec![&c1c, &c2c]);
    for i in 0..result.data.len() {
        println!("{} - {} = {}", c1c.data[i], c2c.data[i], result.data[i]);
        assert!(relative_eq!(result.data[i], c1c.data[i] - c2c.data[i], epsilon = f64::EPSILON));
    }
}
#[cfg(feature = "cuda")]
#[test]
fn forward_cuda() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let mut c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CUDA, false);
    let mut c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CUDA, false);
    let mut result = SubOp::forward(&vec![&c1c, &c2c]);
    assert_eq!(result.data[0], 9.0);

    c1c = Tensor::<f32>::new(&db, vec![15.0, 4.1, 2.3, 34.1, 12.2], vec![2,2], Device::CUDA, false); 
    c2c = Tensor::<f32>::new(&db, vec![6.0, 3.1, 1.3, 4.1, 2.2], vec![2,2], Device::CUDA, false);
    result = SubOp::forward(&vec![&c1c, &c2c]);
    for i in 0..result.data.len() {
        assert!(relative_eq!(result.data[i], c1c.data[i] - c2c.data[i], epsilon = f32::EPSILON));
    }
}


#[test]
fn backward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    let c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
    let c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
    let r1 = MulOp::forward(&vec![&x, &c1c]);
    let r2 = MulOp::forward(&vec![&x, &c2c]);
    let mut result = SubOp::forward(&vec![&r1, &r2]);
    result = MulOp::forward(&vec![&result, &x]);
    assert!(relative_eq!(result.data[0], 225.0));

    let grad = result.backward(None);
    assert!(relative_eq!(grad.get(&x.id).unwrap().data[0], 90.0));
}
