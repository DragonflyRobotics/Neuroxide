use std::sync::{Arc, RwLock};
use approx::relative_eq;
use neuroxide::ops::add::AddOp;
use neuroxide::ops::mul::MulOp;
use neuroxide::ops::op_generic::Operation;
use neuroxide::{ops::matmul::MatMulOp, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};

fn main() {
    let mut db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let a = Tensor::<f32>::new(
        &db,
        vec![
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            // 1.0,
            // 2.0,
            // 3.0,
            // 4.0,
            // 5.0,
            // 6.0,
        ],
        vec![1, 2, 3],
        Device::CPU,
        true,
    );
    let b = Tensor::<f32>::new(
        &db,
        vec![
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            6.0,
            7.0,
            8.0,
            9.0,
            10.0,
            11.0,
            12.0,
            // 13.0,
            // 14.0,
            // 15.0,
            // 16.0,
            // 17.0,
            // 18.0,
            // 19.0,
            // 20.0,
            // 21.0,
            // 22.0,
            // 23.0,
            // 24.0,
            // 25.0,
            // 26.0,
            // 27.0,
            // 28.0,
            // 29.0,
            // 30.0,
            // 31.0,
            // 32.0,
            // 33.0,
            // 34.0,
            // 35.0,
            // 36.0,
        ],
        vec![2, 3, 2],
        Device::CPU,
        true,
    );

    let mut c = MatMulOp::forward(&vec![&a, &b]);
    let grad = c.backward(None);

    println!("Grad: {}", grad.get(&a.id).unwrap());
    println!("Grad: {}", grad.get(&b.id).unwrap());


    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![1, 3, 2], Device::CPU, true);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![11, 15, 19, 11, 15, 19];
    let answer2 = vec![5, 5, 7, 7, 9, 9];
    let grad = c.backward(Some(vec![x.id, x2.id]));
    assert_eq!(grad.get(&x.id).unwrap().data, answer);
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![1, 2, 3]);
    assert_eq!(grad.get(&x2.id).unwrap().data, answer2);
    assert_eq!(grad.get(&x2.id).unwrap().shape, vec![1, 3, 2]);

    
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![3, 2], Device::CPU, true);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let answer = vec![11, 15, 19, 11, 15, 19];
    let answer2 = vec![5, 5, 7, 7, 9, 9];
    let grad = c.backward(Some(vec![x.id, x2.id]));
    assert_eq!(grad.get(&x.id).unwrap().data, answer);
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![2, 3]);
    assert_eq!(grad.get(&x2.id).unwrap().data, answer2);
    assert_eq!(grad.get(&x2.id).unwrap().shape, vec![3, 2]);

    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![6], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![6], Device::CPU, true);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let grad = c.backward(None);
    assert_eq!(grad.get(&x.id).unwrap().data, x2.data);
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![6]);
    assert_eq!(grad.get(&x2.id).unwrap().data, x.data);
    assert_eq!(grad.get(&x2.id).unwrap().shape, vec![6]);

    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7], vec![3], Device::CPU, true);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let grad = c.backward(None);
    let actual_grad = vec![5, 6, 7, 5, 6, 7];
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![1, 2, 3]);
    assert_eq!(grad.get(&x.id).unwrap().data, actual_grad);
    assert_eq!(grad.get(&x2.id).unwrap().shape, vec![1, 3, 1]);
    assert_eq!(grad.get(&x2.id).unwrap().data, vec![5, 7, 9]);
}
