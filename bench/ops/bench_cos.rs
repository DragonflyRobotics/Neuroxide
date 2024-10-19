extern crate test;

use std::sync::{Arc, RwLock};

use neuroxide::{ops::{op_generic::Operation as _, cos::CosOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};

#[bench]
fn forward(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::<f64>::new(&db, vec![0.0, 3.14/6.0, 3.14/4.0, 3.14/3.0, 3.14], vec![1], Device::CPU, false);

    b.iter(|| {
        let _result = CosOp::forward(&vec![&x]);
    });
}

#[cfg(feature = "cuda")]
#[bench]
fn forward_cuda(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x = Tensor::<f32>::new(&db, vec![0.0, 3.14/6.0, 3.14/4.0, 3.14/3.0, 3.14], vec![1], Device::CUDA, false);

    b.iter(|| {
        let _result = CosOp::forward(&vec![&x]);
    });
}


#[bench]
fn backward(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::<f64>::new(&db, vec![0.0, 3.14/6.0, 3.14/4.0, 3.14/3.0, 3.14], vec![1], Device::CPU, true);
    let result = CosOp::forward(&vec![&x]);

    b.iter(|| {
        let _grad = result.backward(None);
    });
}
