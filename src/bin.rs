use std::sync::{Arc, RwLock};

use neuroxide::{ops::{matmul::MatMulOp, pow::PowOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}, utils::array_utils::broadcast_shapes_matmul};
use neuroxide::ops::op_generic::Operation;

#[macro_use]
extern crate neuroxide;


fn main() {
    // let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    // let layer_1_weights = Tensor::<f32>::new(&db, vec![1.0; 16], vec![1, 16], Device::CPU, true);
    // let layer_1_biases = Tensor::<f32>::new(&db, vec![1.0; 16*16], vec![16, 16], Device::CPU, true);
    // let layer_2_weights = Tensor::<f32>::new(&db, vec![1.0; 16], vec![16, 1], Device::CPU, true);
    // let layer_2_biases = Tensor::<f32>::new(&db, vec![1.0; 16], vec![16], Device::CPU, true);
    // let pow_const = Tensor::<f32>::new(&db, vec![2.0; 16], vec![16], Device::CPU, false);
    // let input = Tensor::<f32>::new(&db, vec![1.0; 16], vec![16, 1], Device::CPU, false);
    // let output = Tensor::<f32>::new(&db, vec![2.0; 16], vec![16], Device::CPU, false);
    // let c = matmul!(input, layer_1_weights) + layer_1_biases;
    //
    // let c = matmul!(c, layer_2_weights) + layer_2_biases;
    //
    // let loss = pow!(c - output, pow_const);
    // println!("{}", loss);
    // let grad = loss.backward(None);
    // for i in grad.into_iter() {
    //     println!("{:?}", i.1.shape);
    // }

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7], vec![3], Device::CPU, false);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    println!("{}", c);
    let grad = c.backward(None);
    for i in grad.into_iter() {
        println!("{}", i.1);
    }
}
