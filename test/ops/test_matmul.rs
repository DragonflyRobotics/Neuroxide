use std::sync::{Arc, RwLock};

use neuroxide::{
    ops::{matmul::MatMulOp, op_generic::Operation},
    types::{
        device::Device,
        tensor::Tensor,
        tensordb::{DTypes, TensorDB},
    },
};

#[test]
fn forward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(
        &db,
        vec![1, 2, 3, 4, 5, 6],
        vec![1, 2, 3],
        Device::CPU,
        true,
    );
    let x2 = Tensor::new(
        &db,
        vec![5, 6, 7, 8, 9, 10],
        vec![1, 3, 2],
        Device::CPU,
        false,
    );

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![46, 52, 109, 124];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![1, 2, 2]);

    let x = Tensor::new(&db, vec![1, 2, 3, 4, 5, 6], vec![2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![3, 2], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![46, 52, 109, 124];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![2, 2]);

    let x = Tensor::new(&db, vec![1, 2, 3, 4, 5, 6], vec![6], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![6], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![175];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![1]);

    let a = Tensor::<i32>::new(
        &db,
        vec![1, 2, 3, 4, 5, 6],
        vec![1, 2, 3],
        Device::CPU,
        true,
    );
    let b = Tensor::<i32>::new(
        &db,
        vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        ],
        vec![6, 3, 2],
        Device::CPU,
        true,
    );

    let c = MatMulOp::forward(&vec![&a, &b]);
    let answer = vec![
        22, 28, 49, 64, 58, 64, 139, 154, 94, 100, 229, 244, 130, 136, 319, 334, 166, 172, 409,
        424, 202, 208, 499, 514,
    ];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![6, 2, 2]);
}

#[test]
fn forward_macros() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(
        &db,
        vec![1, 2, 3, 4, 5, 6],
        vec![1, 2, 3],
        Device::CPU,
        true,
    );
    let x2 = Tensor::new(
        &db,
        vec![5, 6, 7, 8, 9, 10],
        vec![1, 3, 2],
        Device::CPU,
        false,
    );

    let c = matmul![&x, &x2];
    let answer = vec![46, 52, 109, 124];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![1, 2, 2]);
}

#[cfg(feature = "cuda")]
#[test]
fn forward_cuda() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x = Tensor::<f32>::new(&db,vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3], Device::CUDA, true);
    let x2 = Tensor::<f32>::new(&db, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![1, 3, 2], Device::CUDA, false);

    let mut c = MatMulOp::forward(&vec![&x, &x2]);
    c.cpu();
    let answer = vec![46.0, 52.0, 109.0, 124.0];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![1, 2, 2]);

    let x = Tensor::<f32>::new(
        &db,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![2, 3],
        Device::CUDA,
        true,
    );
    let x2 = Tensor::<f32>::new(
        &db,
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        vec![3, 2],
        Device::CUDA,
        false,
    );

    let mut c = MatMulOp::forward(&vec![&x, &x2]);
    c.cpu();
    let answer = vec![46.0, 52.0, 109.0, 124.0];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![2, 2]);

    let x = Tensor::<f32>::new(
        &db,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![6],
        Device::CUDA,
        true,
    );
    let x2 = Tensor::<f32>::new(
        &db,
        vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        vec![6],
        Device::CUDA,
        false,
    );

    let mut c = MatMulOp::forward(&vec![&x, &x2]);
    c.cpu();
    let answer = vec![175.0];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![1]);

    let a = Tensor::<f32>::new(
        &db,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![1, 2, 3],
        Device::CUDA,
        true,
    );
    let b = Tensor::<f32>::new(
        &db,
        vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
            17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0,
            32.0, 33.0, 34.0, 35.0, 36.0,
        ],
        vec![6, 3, 2],
        Device::CUDA,
        true,
    );

    let mut c = MatMulOp::forward(&vec![&a, &b]);
    c.cpu();
    let answer = vec![
        22.0, 28.0, 49.0, 64.0, 58.0, 64.0, 139.0, 154.0, 94.0, 100.0, 229.0, 244.0, 130.0, 136.0,
        319.0, 334.0, 166.0, 172.0, 409.0, 424.0, 202.0, 208.0, 499.0, 514.0,
    ];
    assert_eq!(c.data, answer);
    assert_eq!(c.shape, vec![6, 2, 2]);
}

#[test]
fn backward() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(
        &db,
        vec![1, 2, 3, 4, 5, 6],
        vec![1, 2, 3],
        Device::CPU,
        true,
    );
    let x2 = Tensor::new(
        &db,
        vec![5, 6, 7, 8, 9, 10],
        vec![1, 3, 2],
        Device::CPU,
        true,
    );

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![11, 15, 19, 11, 15, 19];
    let answer2 = vec![5, 5, 7, 7, 9, 9];
    let grad = c.backward(Some(vec![x.id, x2.id]));
    assert_eq!(grad.get(&x.id).unwrap().data, answer);
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![1, 2, 3]);
    assert_eq!(grad.get(&x2.id).unwrap().data, answer2);
    assert_eq!(grad.get(&x2.id).unwrap().shape, vec![1, 3, 2]);

    let x = Tensor::new(&db, vec![1, 2, 3, 4, 5, 6], vec![2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![3, 2], Device::CPU, true);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![11, 15, 19, 11, 15, 19];
    let answer2 = vec![5, 5, 7, 7, 9, 9];
    let grad = c.backward(Some(vec![x.id, x2.id]));
    assert_eq!(grad.get(&x.id).unwrap().data, answer);
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![2, 3]);
    assert_eq!(grad.get(&x2.id).unwrap().data, answer2);
    assert_eq!(grad.get(&x2.id).unwrap().shape, vec![3, 2]);

    let x = Tensor::new(&db, vec![1, 2, 3, 4, 5, 6], vec![6], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![6], Device::CPU, true);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let grad = c.backward(None);
    assert_eq!(grad.get(&x.id).unwrap().data, x2.data);
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![6]);
    assert_eq!(grad.get(&x2.id).unwrap().data, x.data);
    assert_eq!(grad.get(&x2.id).unwrap().shape, vec![6]);

    let x = Tensor::new(
        &db,
        vec![1, 2, 3, 4, 5, 6],
        vec![1, 2, 3],
        Device::CPU,
        true,
    );
    let x2 = Tensor::new(&db, vec![5, 6, 7], vec![3], Device::CPU, true);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let grad = c.backward(None);
    let actual_grad = vec![5, 6, 7, 5, 6, 7];
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![1, 2, 3]);
    assert_eq!(grad.get(&x.id).unwrap().data, actual_grad);
    assert_eq!(grad.get(&x2.id).unwrap().shape, vec![1, 3, 1]);
    assert_eq!(grad.get(&x2.id).unwrap().data, vec![5, 7, 9]);

    let a = Tensor::<i32>::new(
        &db,
        vec![1, 2, 3, 4, 5, 6],
        vec![1, 2, 3],
        Device::CPU,
        true,
    );
    let b = Tensor::<i32>::new(
        &db,
        vec![
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36,
        ],
        vec![6, 3, 2],
        Device::CPU,
        true,
    );

    let c = MatMulOp::forward(&vec![&a, &b]);
    let grad = c.backward(None);
    let actual_grad_a = vec![198, 222, 246, 198, 222, 246];
    assert_eq!(grad.get(&a.id).unwrap().data, actual_grad_a);
    assert_eq!(grad.get(&a.id).unwrap().shape, vec![1, 2, 3]);
    let actual_grad_b = vec![
        5, 5, 7, 7, 9, 9, 5, 5, 7, 7, 9, 9, 5, 5, 7, 7, 9, 9, 5, 5, 7, 7, 9, 9, 5, 5, 7, 7, 9, 9,
        5, 5, 7, 7, 9, 9,
    ];
    assert_eq!(grad.get(&b.id).unwrap().data, actual_grad_b);
    assert_eq!(grad.get(&b.id).unwrap().shape, vec![6, 3, 2]);
}
