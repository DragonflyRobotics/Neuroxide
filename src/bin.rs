use std::{sync::{Arc, RwLock}, time::{SystemTime, UNIX_EPOCH}};

use ndarray::{ArrayD, IxDyn};
use neuroxide::{ops::{matmul::MatMulOp, mul::MulOp, pow::PowOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}, utils::array_utils::broadcast_shapes_matmul};
use neuroxide::ops::op_generic::Operation;
use petgraph::dot::{Config, Dot};

#[macro_use]
extern crate neuroxide;


fn main() {
    // let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    // for i in 0..10000{
    //     let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    //     let layer_1_weights = Tensor::<f32>::new(&db, vec![1.0; 16], vec![1, 16], Device::CUDA, true);
    //     let layer_1_biases = Tensor::<f32>::new(&db, vec![1.0; 16*16], vec![16, 16], Device::CUDA, true);
    //     let layer_2_weights = Tensor::<f32>::new(&db, vec![1.0; 16], vec![16, 1], Device::CUDA, true);
    //     let layer_2_biases = Tensor::<f32>::new(&db, vec![1.0; 16], vec![16, 1], Device::CUDA, true);
    //     let pow_const = Tensor::<f32>::new(&db, vec![2.0; 16], vec![16, 1], Device::CUDA, false);
    //     let input = Tensor::<f32>::new(&db, vec![1.0; 16], vec![16, 1], Device::CUDA, false);
    //     let output = Tensor::<f32>::new(&db, vec![2.0; 16], vec![16, 1], Device::CUDA, false);
    //     let c = matmul!(input, layer_1_weights) + layer_1_biases;
        
    //     let c = matmul!(c, layer_2_weights) + layer_2_biases.clone();
        
        
    //     let loss = pow!(c - output, pow_const);
    //     let grad = loss.backward(None);
    //     println!("{:?}", i);
    // }
    // let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap(); 
    // // for i in grad.into_iter() {
    // //     println!("{}", i.1);
    // // }
    // println!("Time taken: {:?}", end - start);
    // // println!("{}", grad.get(&layer_2_biases.id).unwrap());
    // // println!("{:?}", Dot::with_config(&loss.op_chain, &[Config::EdgeNoLabel]));
    
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    
    let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6], vec![1, 2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5, 6, 7], vec![3], Device::CPU, true);

    let c = MatMulOp::forward(&vec![&x, &x2]);
    let grad = c.backward(None);
    let actual_grad = vec![5, 6, 7, 5, 6, 7];
    println!("{:?}", grad.get(&x2.id).unwrap().shape);
    assert_eq!(grad.get(&x.id).unwrap().shape, vec![1, 2, 3]);
    assert_eq!(grad.get(&x.id).unwrap().data, actual_grad);
    assert_eq!(grad.get(&x2.id).unwrap().shape, vec![3]);
    assert_eq!(grad.get(&x2.id).unwrap().data, vec![5, 7, 9]);
}
