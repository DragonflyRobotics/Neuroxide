use std::{sync::{Arc, RwLock}, time::{SystemTime, UNIX_EPOCH}};

use neuroxide::types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}};
use neuroxide::ops::op_generic::Operation;
use rand::Rng;

#[macro_use]
extern crate neuroxide;

// TODO: Fix benchmarks
// TODO: Replace NDArray (Maybe)
// TODO: Create Union Graph for operations on CUDA
// TODO: Enable caching/saving
// TODO: Add more operations

fn main() {
    let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let mut layer_1_weights = Tensor::<f32>::new_uniform(&db, vec![1, 16], Device::CPU, true);
    let mut layer_1_biases = Tensor::<f32>::new_zeros(&db, vec![16, 16], Device::CPU, true);
    let mut layer_2_weights = Tensor::<f32>::new_uniform(&db, vec![16, 1], Device::CPU, true);
    let mut layer_2_biases = Tensor::<f32>::new_zeros(&db, vec![16, 1], Device::CPU, true);
    let pow_const = Tensor::<f32>::new(&db, vec![2.0; 16], vec![16, 1], Device::CPU, false);
    let lr = Tensor::<f32>::new(&db, vec![0.0000001], vec![1], Device::CPU, false);
    for iteration in 0..600 {
        let num: f32 = rand::thread_rng().gen_range(0..100) as f32; 
        let input = Tensor::<f32>::new(&db, vec![num; 16], vec![16, 1], Device::CPU, false);
        let output = Tensor::<f32>::new(&db, vec![num * 2.0; 16], vec![16, 1], Device::CPU, false);

        let c = add!(matmul!(input, layer_1_weights), layer_1_biases);
        let c = add!(matmul!(c, layer_2_weights), layer_2_biases);
        let loss = pow!(c - output, pow_const);

        let grad = loss.backward(None);

        layer_1_weights = layer_1_weights.clone() - grad.get(&layer_1_weights.id).unwrap().clone() * lr.clone();
        layer_1_weights.clear_graph();

        layer_1_biases = layer_1_biases.clone() - grad.get(&layer_1_biases.id).unwrap().clone() * lr.clone();
        layer_1_biases.clear_graph();

        layer_2_weights = layer_2_weights.clone() - grad.get(&layer_2_weights.id).unwrap().clone() * lr.clone();
        layer_2_weights.clear_graph();

        layer_2_biases = layer_2_biases.clone() - grad.get(&layer_2_biases.id).unwrap().clone() * lr.clone();
        layer_2_biases.clear_graph();
        println!("Epoch: {} Loss: {}", iteration, loss);
    } 

    let input = Tensor::<f32>::new(&db, vec![4.0; 16], vec![16, 1], Device::CPU, false);

    let c = add!(matmul!(input, layer_1_weights), layer_1_biases);
    let c = add!(matmul!(c, layer_2_weights), layer_2_biases);
    println!("{}", c);


    // let input = Tensor::<f32>::new(&db, vec![400.0; 16], vec![16, 1], Device::CPU, false);
    //
    // let c = add!(matmul!(input, layer_1_weights), layer_1_biases);
    // let c = add!(matmul!(c, layer_2_weights), layer_2_biases);
    // println!("{}", c);


    let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("Time taken: {:?} seconds", end-start);
}


// fn main() {
//     //make two arrays 16000 x16000 and multiply them
//     let a = vec![2.0; 16000*16000];
//     let a_arr = ndarray::Array2::from_shape_vec((16000, 16000), a).unwrap();
//     let b = vec![4.0; 16000*16000];
//     let b_arr = ndarray::Array2::from_shape_vec((16000, 16000), b).unwrap();
//
//     let start = std::time::Instant::now();
//     let c = a_arr.dot(&b_arr);
//     let end = std::time::Instant::now();
//     println!("Time taken: {:?}", end-start);
// }
