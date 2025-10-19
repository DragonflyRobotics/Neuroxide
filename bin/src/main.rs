use std::{sync::{Arc, RwLock}, time::{SystemTime, UNIX_EPOCH}};

use approx::relative_eq;
use neuroxide::{layers::linear::Linear, ops::{add::AddOp, matmul::MatMulOp, mul::MulOp, sub::SubOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use neuroxide::ops::op_generic::Operation;
use neuroxide::optimizers::simple_descent::SimpleDescent;
use rand::Rng;


#[cfg(feature = "cuda")]
unsafe extern "C" {
pub fn transpose(m: i32, n: i32, a: *mut f32, c: *mut*mut f32) -> CudnnStatusT;
pub fn transpose_b(d: i32, m: i32, n: i32, broad: i32, a: *mut f32, c: *mut*mut f32) -> CudnnStatusT;
pub fn toCpu(size: i32, ptr: *mut f32) -> *mut f32;
pub fn toCuda(size: i32, data: *mut f32) -> *mut f32;
pub fn reduce(d: i32, m: i32, n: i32, a: *mut f32, c: *mut*mut f32) -> CudnnStatusT;
}

pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses

#[macro_use]
extern crate neuroxide;

// TODO: Fix benchmarks
// TODO: Replace NDArray (Maybe)
// TODO: Create Union Graph for operations on CUDA
// TODO: Enable caching/saving
// TODO: Add more operations

fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let pow_const = Tensor::<f32>::new(&db, vec![2.0; 16], vec![1,16], Device::CUDA, false);
    let mut linear1 = Linear::new(&db, 16, 16, true);
    let mut linear2 = Linear::new(&db, 16, 16, true);
    let mut optim = SimpleDescent::new(&db, 0.0000001);
    optim.add_parameters(&linear1.parameters());
    optim.add_parameters(&linear2.parameters());
    let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    for _ in 0..1500 {
        let num: f32 = rand::rng().random_range(0..100) as f32; 
        let input = Tensor::<f32>::new(&db, vec![num; 16], vec![1, 16], Device::CUDA, false);
        let output = Tensor::<f32>::new(&db, vec![num * 2.0; 16], vec![1, 16], Device::CUDA, false);


        let mut c = linear1.forward(&input);
        c = linear2.forward(&c);

        let loss = pow!(c - output, pow_const);
        let grad = loss.backward(None, Device::CUDA);

        optim.step(&grad);

    } 

    let input = Tensor::<f32>::new(&db, vec![4.0; 16], vec![1, 16], Device::CUDA, false);

    let c = linear1.forward(&input);
    let mut c = linear2.forward(&c);
    c.cpu();
    println!("{}", c);


    let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("Time taken: {:?} seconds", end-start);
}

// fn main() {
//     // let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
//     let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
//     // let mut x = Tensor::<f32>::new(&db, vec![5.0], vec![1], Device::CUDA, true);
//     // let c1c = Tensor::<f32>::new(&db, vec![15.0], vec![1], Device::CUDA, false);
//     // let c2c = Tensor::<f32>::new(&db, vec![6.0], vec![1], Device::CUDA, false);
//     // let r1 = MulOp::forward(&vec![&x, &c1c]);
//     // let r2 = MulOp::forward(&vec![&x, &c2c]);
//     // let mut result = AddOp::forward(&vec![&r1, &r2]);
//     // result = MulOp::forward(&vec![&result, &x]);
//     // // assert!(relative_eq!(result.data[0], 525.0));
//     // // assert_eq!(result.shape, vec![1]);
//     //
//     // let grad = result.backward(None, Device::CUDA);
//     // let mut x_grad = grad.get(&x.id).unwrap().clone();
//     // x = SubOp::forward(&vec![&x, &x_grad]);
//     // x.clear_graph();
//     //
//     //
//     // let r1 = MulOp::forward(&vec![&x, &c1c]);
//     // let r2 = MulOp::forward(&vec![&x, &c2c]);
//     // let mut result = AddOp::forward(&vec![&r1, &r2]);
//     // result = MulOp::forward(&vec![&result, &x]);
//     // let grad = result.backward(None, Device::CUDA);
//     // let mut x_grad = grad.get(&x.id).unwrap().clone();
//     // x_grad.cpu();
//     // let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
//     // println!("x_grad: {}", x_grad.data[0]);
//     // println!("Time taken: {:?} seconds", end - start);
//     //
//     // let d: i32 = 2;
//     // let m: i32 = 2;
//     // let n: i32 = 3;
//     // let mut a: Vec<f32> = vec![
//     //     1.0,
//     //     2.0,
//     //     3.0,
//     //     4.0,
//     //     5.0,
//     //     6.0,
//     //     1.0,
//     //     2.0,
//     //     3.0,
//     //     4.0,
//     //     5.0,
//     //     6.0,
//     // ];
//     // let mut data: f32 = 0.0;
//     // let mut ptr_to_data: *mut f32 = &mut data;
//     // let mut r = vec![0.0; m as usize * n as usize];
//     // #[cfg(feature = "cuda")]
//     // unsafe {
//     //     let cuda_ptr = toCuda(a.len() as i32, a.as_mut_ptr() as *mut f32);
//     //     reduce(d, m, n, cuda_ptr, &mut ptr_to_data);
//     //     // transpose_b(d as i32, m as i32, n as i32, d, cuda_ptr, &mut ptr_to_data);
//     //     let ptr = toCpu(m*n, ptr_to_data);
//     //     //
//     //     std::ptr::copy_nonoverlapping(ptr, r.as_mut_ptr(), m as usize * n as usize);
//     //     println!("{:?}", r);
//     // }
//     let a = Tensor::<f32>::new(
//         &db,
//         vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
//         vec![1, 2, 3],
//         Device::CUDA,
//         true,
//     );
//     let b = Tensor::<f32>::new(
//         &db,
//         vec![
//             1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
//             25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0,
//         ],
//         vec![6, 3, 2],
//         Device::CUDA,
//         true,
//     );
//
//     let c = MatMulOp::forward(&vec![&a, &b]);
//     let grad = c.backward(None, Device::CUDA);
//     let mut c_grad = grad.get(&a.id).unwrap().clone();
//     c_grad.cpu();
//     println!("c: {}", c_grad);
//
//     let actual_grad_a = vec![198, 222, 246, 198, 222, 246];
// }
//
