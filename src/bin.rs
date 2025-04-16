extern crate blas_src;
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
    // blas_src::blas_set_num_threads(4);
    let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let mut layer_1_weights = Tensor::<f32>::new_uniform(&db, vec![16, 32], Device::CPU, true);
    let mut layer_1_biases = Tensor::<f32>::new_zeros(&db, vec![32], Device::CPU, true);
    let mut layer_2_weights = Tensor::<f32>::new_uniform(&db, vec![32, 16], Device::CPU, true);
   let mut layer_2_biases = Tensor::<f32>::new_zeros(&db, vec![16], Device::CPU, true);
    let pow_const = Tensor::<f32>::new(&db, vec![2.0; 1], vec![1], Device::CPU, false);
    let lr = Tensor::<f32>::new(&db, vec![0.0000001], vec![1], Device::CPU, false);
    for iteration in 0..600 {
        let num: f32 = rand::thread_rng().gen_range(0..100) as f32; 
        let input = Tensor::<f32>::new(&db, vec![num; 16], vec![1, 16], Device::CPU, false);
        let output = Tensor::<f32>::new(&db, vec![num * 2.0; 16], vec![1, 16], Device::CPU, false);


        let mut c = matmul!(input, layer_1_weights);
        // println!("c: {:?}", c.shape);
        c = add!(c, layer_1_biases);
        // println!("c: {:?}", c.shape);
        c = matmul!(c, layer_2_weights);
        // println!("c: {:?}", c.shape);
        c = add!(c, layer_2_biases);
        // println!("c: {:?}", c.shape);

        let loss = pow!(c - output, pow_const);
        // println!("loss: {:?}", loss.shape);

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

    let input = Tensor::<f32>::new(&db, vec![4.0; 16], vec![16], Device::CPU, false);

    let c = add!(matmul!(input, layer_1_weights), layer_1_biases);
    let c = add!(matmul!(c, layer_2_weights), layer_2_biases);
    println!("{}", c);


    let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("Time taken: {:?} seconds", end-start);
}
