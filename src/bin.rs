extern crate blas_src;
use std::{sync::{Arc, RwLock}, time::{SystemTime, UNIX_EPOCH}};

use libc::free;
use neuroxide::{layers::linear::Linear, ops::matmul::MatMulOp, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
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

#[cfg(feature = "cuda")]
extern "C" {
pub fn matmul(m: i32, n: i32, k: i32, h_A: *mut f32, h_B: *mut f32, h_C: *mut*mut f32) -> CudnnStatusT;
fn checkkData(len: i32, ptr: *mut f32) -> i32;
fn destroyPool();
fn createPoolMax();
}
pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses
//
// fn main() {
//     let mut db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
//     let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
//     for i in 0..20 {
//         let a = Tensor::<f32>::new_uniform(&db, vec![4000, 4000], Device::CUDA, true);
//         let b = Tensor::<f32>::new_uniform(&db, vec![4000, 4000], Device::CUDA, false);
//         let mut c = MatMulOp::forward(&vec![&a, &b]);
//         c.backward(None);
//         db.write().unwrap().clear();
//
//         // unsafe {
//         //     destroyPool();
//         //     createPoolMax();
//         // }
//     }
//     println!("dfon3");
//     let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
//     println!("Time taken: {:?} seconds", end-start);
// }



fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let pow_const = Tensor::<f32>::new(&db, vec![2.0; 16], vec![1,16], Device::CUDA, false);
    let mut linear1 = Linear::new(&db, 16, 16, true);
    let mut linear2 = Linear::new(&db, 16, 16, true);
    let mut optim = SimpleDescent::new(&db, 0.0000001);
    optim.add_parameters(&linear1.parameters());
    optim.add_parameters(&linear2.parameters());
    let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    for iteration in 0..1500 {
        let num: f32 = rand::thread_rng().gen_range(0..100) as f32; 
        let input = Tensor::<f32>::new(&db, vec![num; 16], vec![1, 16], Device::CUDA, false);
        let output = Tensor::<f32>::new(&db, vec![num * 2.0; 16], vec![1, 16], Device::CUDA, false);


        let mut c = linear1.forward(&input);
        c = linear2.forward(&c);

        let loss = pow!(c - output, pow_const);
        let grad = loss.backward(None);

        optim.step(&grad);

        // println!("Epoch: {} Loss: {}", iteration, loss);
    } 

    let input = Tensor::<f32>::new(&db, vec![4.0; 16], vec![16], Device::CUDA, false);

    let c = linear1.forward(&input);
    let mut c = linear2.forward(&c);
    c.cpu();
    println!("{}", c);


    let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("Time taken: {:?} seconds", end-start);
}
