use std::sync::{Arc, RwLock};

use approx::relative_eq;
use neuroxide::{ops::{op_generic::Operation, pow::PowOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};


#[test]
fn forward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let mut c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
    let mut c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
    let mut result = PowOp::forward(&vec![&c1c, &c2c]);
    assert_eq!(result.data[0], 15.0_f64.powf(6.0));

    c1c = Tensor::new(&db, vec![15.0, 4.1, 2.3, 34.1, 12.2], vec![2,2], Device::CPU, false); 
    c2c = Tensor::new(&db, vec![6.0, 3.1, 1.3, 4.1, 2.2], vec![2,2], Device::CPU, false);
    result = PowOp::forward(&vec![&c1c, &c2c]);
    for i in 0..result.data.len() {
        assert!(relative_eq!(result.data[i], c1c.data[i].powf(c2c.data[i]), epsilon = f64::EPSILON));
    }
}

#[cfg(feature = "cuda")]
#[test]
fn forward_cuda() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let mut c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CUDA, false);
    let mut c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CUDA, false);
    let mut result = PowOp::forward(&vec![&c1c, &c2c]);
    assert_eq!(result.data[0], 15.0_f32.powf(6.0));

    c1c = Tensor::<f32>::new(&db, vec![15.0, 4.1, 2.3, 34.1, 12.2], vec![2,2], Device::CUDA, false); 
    c2c = Tensor::<f32>::new(&db, vec![6.0, 3.1, 1.3, 4.1, 2.2], vec![2,2], Device::CUDA, false);
    result = PowOp::forward(&vec![&c1c, &c2c]);
    for i in 0..result.data.len() {
        assert!(relative_eq!(result.data[i], c1c.data[i].powf(c2c.data[i]), epsilon = f32::EPSILON));
    }
}


#[test]
fn backward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    let c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, true);

    let result = PowOp::forward(&vec![&x, &c1c]);
    let grad = result.backward(None);
    let grad_x = &grad[&x.id].data;
    let grad_c1c = &grad[&c1c.id].data;
    assert_eq!(grad_x[0], 15.0_f64 * 5.0_f64.powf(15.0_f64 - 1.0));
    assert_eq!(grad_c1c[0], 5.0_f64.powf(15.0_f64) * (5.0_f64.ln()));
    
}
