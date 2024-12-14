use std::sync::{Arc, RwLock};

use neuroxide::{ops::{matmul::MatMulOp, op_generic::Operation}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};


#[test]
fn forward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![1, 3, 2], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![46, 52, 109, 124];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![1, 2, 2]);


    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![3, 2], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![46, 52, 109, 124];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![2, 2]);


    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![6], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![6], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![175];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![1]);

}

#[test]
fn forward_macros() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![1, 3, 2], Device::CPU, false);

    let c = matmul![&x, &x2];
    let answer = vec![46, 52, 109, 124];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![1, 2, 2]);
}


#[test]
fn backward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![1, 3, 2], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![11, 15, 19, 11, 15, 19];
    let grad = c.backward(Some(vec![x.id]));
    assert_eq!(grad.get(&x.id).unwrap().data, answer);
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![1, 2, 3]);

    
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![3, 2], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![11, 15, 19, 11, 15, 19];
    let grad = c.backward(Some(vec![x.id]));
    assert_eq!(grad.get(&x.id).unwrap().data, answer);
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![2, 3]);

    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![6], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![6], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let grad = c.backward(None);
    assert_eq!(grad.get(&x.id).unwrap().data, x2.data);
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![6]);

}
