use std::sync::{Arc, RwLock};

use approx::relative_eq;
use neuroxide::{ops::{add::AddOp, matmul::MatMulOp, mul::MulOp, pow::PowOp, sub::SubOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use neuroxide::ops::op_generic::Operation;



#[cfg(feature = "cuda")]
extern "C" {
pub fn matmul(m: i32, n: i32, k: i32, h_A: *mut f32, h_B: *mut f32, h_C: *mut f32) -> CudnnStatusT;
}

pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses


#[macro_use]
extern crate neuroxide;


fn main() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let a_shape = vec![2, 2, 2];
    let b_shape = vec![2, 2, 3];

    let tensor_db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let c1 = Tensor::new(&tensor_db, vec![6.0; 12], vec![2, 2, 3], Device::CUDA, true);
    let c2 = Tensor::new(&tensor_db, vec![5.0; 12], vec![2, 2, 3], Device::CUDA, true);
    let c3 = Tensor::new(&tensor_db, vec![2.0; 12], vec![2, 2, 3], Device::CUDA, true);


    let a = Tensor::new(&tensor_db, a, a_shape, Device::CUDA, true);
    let b = Tensor::new(&tensor_db, b, b_shape, Device::CUDA, false);
    let c = MatMulOp::forward(&vec![&a, &b]);
    let d = MulOp::forward(&vec![&c, &c1]);
    let e = AddOp::forward(&vec![&d, &c2]);
    let f = PowOp::forward(&vec![&e, &c3]);
    let grad = f.backward(None);
    println!("{}", grad.get(&a.id).unwrap());

}
