extern crate test;

use std::sync::{Arc, RwLock};

use neuroxide::{ops::{matmul::MatMulOp, op_generic::Operation}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};


#[bench]
fn forward(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![1, 3, 2], Device::CPU, false);

    b.iter(|| {
        let _result = MatMulOp::forward(&vec![&x, &x2]);
    });


    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![3, 2], Device::CPU, false);

    b.iter(|| {
        let _result = MatMulOp::forward(&vec![&x, &x2]);
    });

    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![6], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![6], Device::CPU, false);

    b.iter(|| {
        let _result = MatMulOp::forward(&vec![&x, &x2]);
    });
}

#[cfg(feature = "cuda")]
#[bench]
fn forward_cuda(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x = Tensor::<f32>::new(&db,vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3], Device::CUDA, true);
    let x2 = Tensor::<f32>::new(&db, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![1, 3, 2], Device::CUDA, false);

    b.iter(|| {
        let _result = MatMulOp::forward(&vec![&x, &x2]);
    });

    let x = Tensor::<f32>::new(&db,vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Device::CUDA, true);
    let x2 = Tensor::<f32>::new(&db, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![3, 2], Device::CUDA, false);

    b.iter(|| {
        let _result = MatMulOp::forward(&vec![&x, &x2]);
    });

    let x = Tensor::<f32>::new(&db,vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6], Device::CUDA, true);
    let x2 = Tensor::<f32>::new(&db, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![6], Device::CUDA, false);

    b.iter(|| {
        let _result = MatMulOp::forward(&vec![&x, &x2]);
    });
}


#[bench]
fn backward(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![1, 3, 2], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);

    b.iter(|| {
        let _grad = c.backward(Some(vec![x.id]));
    });

    
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![3, 2], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    b.iter(|| {
        let _grad = c.backward(Some(vec![x.id]));
    });

    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![6], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![6], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    b.iter(|| {
        let _grad = c.backward(Some(vec![x.id]));
    });
}
