use approx::relative_eq;
use neuroxide::ops::add::AddOp;
use neuroxide::ops::ln::LnOp;
use neuroxide::ops::mul::MulOp;
use neuroxide::ops::op_generic::Operation;
use neuroxide::{
    ops::matmul::MatMulOp,
    types::{
        device::Device,
        tensor::Tensor,
        tensordb::{DTypes, TensorDB},
    },
};
use std::sync::{Arc, RwLock};

fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let a = Tensor::<f32>::new(
        &db,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        vec![1, 2, 3],
        Device::CUDA,
        true,
    );
    println!("Tensor a: {}", a);
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
