use std::sync::{Arc, RwLock};

use neuroxide::{ops::{add::AddOp, cos::CosOp, mul::MulOp, op_generic::Operation}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use petgraph::dot::Dot;


// fn main() {
//     let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
//     let x1 = Tensor::new(&db, vec![5i32; 10000000], vec![1], Device::CUDA, true);
//     let x2 = Tensor::new(&db, vec![6i32; 10000000], vec![1], Device::CUDA, false);
//
//     let result = MulOp::forward(&vec![&x1, &x2]);
//     for _ in 0..5 {
//         let result = MulOp::forward(&vec![&x1, &x2]); 
//     }
//     println!("cuda {:?}", result.data[0]);
//     let grad = result.backward(None);
//     println!("{}", grad.get(&x1.id).unwrap().data[0]);
// }

#[cfg(feature = "cuda")]
extern "C" {
pub fn add_kernel(len: i32, a: *mut f64, b: *mut f64, c: *mut f64) -> CudnnStatusT;
}

pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses

#[cfg(not(tarpaulin_include))]
fn main() {
    let mut A: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let mut B: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
    let mut C: Vec<f64> = vec![0.0; 4];
    unsafe {
        add_kernel(4, A.as_mut_ptr(), B.as_mut_ptr(), C.as_mut_ptr());
    }
    println!("{:?}", C);
}
