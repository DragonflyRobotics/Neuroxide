extern crate blas_src;
use std::{sync::{Arc, RwLock}, time::{SystemTime, UNIX_EPOCH}};

use neuroxide::types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}};
use neuroxide::ops::op_generic::Operation;
use neuroxide::ops::matmul::MatMulOp;
use rand::Rng;

#[macro_use]
extern crate neuroxide;

// TODO: Fix benchmarks
// TODO: Replace NDArray (Maybe)
// TODO: Create Union Graph for operations on CUDA
// TODO: Enable caching/saving
// TODO: Add more operations

fn main() {
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
    println!("{:?}", c.shape);
    // let grad = c.backward(None);
    // let actual_grad = vec![5, 6, 7, 5, 6, 7];
    // assert_eq!(grad.get(&x.id).unwrap().shape, vec![1, 2, 3]);
    // assert_eq!(grad.get(&x.id).unwrap().data, actual_grad);
    // assert_eq!(grad.get(&x2.id).unwrap().shape, vec![1, 3, 1]);
    // assert_eq!(grad.get(&x2.id).unwrap().data, vec![5, 7, 9]);

}

// fn main() {
//     // blas_src::blas_set_num_threads(4);
//     let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
//     let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
//     let mut layer_1_weights = Tensor::<f32>::new_uniform(&db, vec![1, 16], Device::CPU, true);
//     let mut layer_1_biases = Tensor::<f32>::new_zeros(&db, vec![16, 16], Device::CPU, true);
//     let mut layer_2_weights = Tensor::<f32>::new_uniform(&db, vec![16, 1], Device::CPU, true);
//     let mut layer_2_biases = Tensor::<f32>::new_zeros(&db, vec![16, 1], Device::CPU, true);
//     let pow_const = Tensor::<f32>::new(&db, vec![2.0; 16], vec![16, 1], Device::CPU, false);
//     let lr = Tensor::<f32>::new(&db, vec![0.0000001], vec![1], Device::CPU, false);
//     for iteration in 0..600 {
//         let num: f32 = rand::thread_rng().gen_range(0..100) as f32; 
//         let input = Tensor::<f32>::new(&db, vec![num; 16], vec![16], Device::CPU, false);
//         let output = Tensor::<f32>::new(&db, vec![num * 2.0; 16], vec![16], Device::CPU, false);

//         let c = add!(matmul!(input, layer_1_weights), layer_1_biases);
//         let c = add!(matmul!(c, layer_2_weights), layer_2_biases);
//         let loss = pow!(c - output, pow_const);

//         let grad = loss.backward(None);

//         layer_1_weights = layer_1_weights.clone() - grad.get(&layer_1_weights.id).unwrap().clone() * lr.clone();
//         layer_1_weights.clear_graph();

//         layer_1_biases = layer_1_biases.clone() - grad.get(&layer_1_biases.id).unwrap().clone() * lr.clone();
//         layer_1_biases.clear_graph();

//         layer_2_weights = layer_2_weights.clone() - grad.get(&layer_2_weights.id).unwrap().clone() * lr.clone();
//         layer_2_weights.clear_graph();

//         layer_2_biases = layer_2_biases.clone() - grad.get(&layer_2_biases.id).unwrap().clone() * lr.clone();
//         layer_2_biases.clear_graph();
//         // println!("Epoch: {} Loss: {}", iteration, loss);
//     } 

//     let input = Tensor::<f32>::new(&db, vec![4.0; 16], vec![16], Device::CPU, false);

//     let c = add!(matmul!(input, layer_1_weights), layer_1_biases);
//     let c = add!(matmul!(c, layer_2_weights), layer_2_biases);
//     println!("{}", c);


//     // let input = Tensor::<f32>::new(&db, vec![400.0; 16], vec![16, 1], Device::CPU, false);
//     //
//     // let c = add!(matmul!(input, layer_1_weights), layer_1_biases);
//     // let c = add!(matmul!(c, layer_2_weights), layer_2_biases);
//     // println!("{}", c);


//     let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
//     println!("Time taken: {:?} seconds", end-start);
// }
