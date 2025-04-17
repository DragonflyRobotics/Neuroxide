extern crate blas_src;
use std::{sync::{Arc, RwLock}, time::{SystemTime, UNIX_EPOCH}};

use neuroxide::{layers::linear::Linear, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use neuroxide::ops::op_generic::Operation;
use neuroxide::optimizers::simple_descent::SimpleDescent;
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
    let pow_const = Tensor::<f32>::new(&db, vec![2.0; 1], vec![1], Device::CPU, false);
    let lr = Tensor::<f32>::new(&db, vec![0.0000001], vec![1], Device::CPU, false);
    let mut linear1 = Linear::new(&db, 16, 16, true);
    let mut optim = SimpleDescent::new(&db, lr.clone());
    optim.add_parameters(&linear1.parameters());
    for iteration in 0..2000 {
        let num: f32 = rand::thread_rng().gen_range(0..100) as f32; 
        let input = Tensor::<f32>::new(&db, vec![num; 16], vec![1, 16], Device::CPU, false);
        let output = Tensor::<f32>::new(&db, vec![num * 2.0; 16], vec![1, 16], Device::CPU, false);


        let c = linear1.forward(&input);

        let loss = pow!(c - output, pow_const);
        // println!("loss: {:?}", loss.shape);

        let grad = loss.backward(None);
        // println!("{}", db.read().unwrap().get(*linear1.parameters().values().nth(0).unwrap()).unwrap()); 

        optim.step(&grad);


        println!("Epoch: {} Loss: {}", iteration, loss);
    } 

    let input = Tensor::<f32>::new(&db, vec![4.0; 16], vec![16], Device::CPU, false);

    let c = linear1.forward(&input);
    println!("{}", c);


    let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("Time taken: {:?} seconds", end-start);
}
