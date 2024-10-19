extern crate test;

use std::sync::{Arc, RwLock};

use neuroxide::{ops::{div::DivOp, op_generic::Operation as _}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};

#[bench]
fn forward(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let mut c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
    let mut c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
    b.iter(|| {
        let _result = DivOp::forward(&vec![&c1c, &c2c]);
    });

    c1c = Tensor::new(&db, vec![15.0, 4.1, 2.3, 34.1, 12.2], vec![2,2], Device::CPU, false); 
    c2c = Tensor::new(&db, vec![6.0, 3.1, 1.3, 4.1, 2.2], vec![2,2], Device::CPU, false);
    b.iter(|| {
        let _result = DivOp::forward(&vec![&c1c, &c2c]);
    });
}

#[cfg(feature = "cuda")]
#[bench]
fn forward_cuda(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let mut c1c = Tensor::<f32>::new(&db, vec![15.0], vec![1], Device::CUDA, false);
    let mut c2c = Tensor::<f32>::new(&db, vec![6.0], vec![1], Device::CUDA, false);
    b.iter(|| {
        let _result = DivOp::forward(&vec![&c1c, &c2c]);
    });

    c1c = Tensor::<f32>::new(&db, vec![15.0, 4.1, 2.3, 34.1, 12.2], vec![2,2], Device::CUDA, false); 
    c2c = Tensor::<f32>::new(&db, vec![6.0, 3.1, 1.3, 4.1, 2.2], vec![2,2], Device::CUDA, false);
    b.iter(|| {
        let _result = DivOp::forward(&vec![&c1c, &c2c]);
    });
}


#[bench]
fn backward(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    let c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, true);
    
    let result = DivOp::forward(&vec![&x, &c1c]);

    b.iter(|| {
        let _grad = result.backward(None);
    });
}
