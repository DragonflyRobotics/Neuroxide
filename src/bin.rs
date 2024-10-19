use std::sync::{Arc, RwLock};

use neuroxide::{ops::{add::AddOp, cos::CosOp, mul::MulOp, op_generic::Operation, sin::SinOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use petgraph::dot::Dot;


// fn main() {
//     let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
//     let x1 = Tensor::new(&db, vec![5.0f32; 10000000], vec![1], Device::CUDA, true);
//     let x2 = Tensor::new(&db, vec![6.0f32; 10000000], vec![1], Device::CUDA, false);
//
//     let result = CosOp::forward(&vec![&x1]);
//     for _ in 0..5 {
//         let result = CosOp::forward(&vec![&x1]); 
//     }
//     println!("cuda {:?}", result.data[0]);
//     let grad = result.backward(None);
//     println!("{}", grad.get(&x1.id).unwrap().data[0]);
// }

// #[cfg(feature = "cuda")]
// extern "C" {
// pub fn sin_kernel(len: i32, a: *mut f32, c: *mut f32) -> CudnnStatusT;
// pub fn cos_kernel(len: i32, a: *mut f32, c: *mut f32) -> CudnnStatusT;
// }
//
// pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses
//
// #[cfg(not(tarpaulin_include))]
// fn main() {
//     let mut A: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
//     let mut C: Vec<f32> = vec![0.0; 4];
//     unsafe {
//         cos_kernel(4, A.as_mut_ptr(), C.as_mut_ptr());
//     }
//     println!("{:?}", C);
// }

fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x1 = Tensor::<f32>::new(&db, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 4], Device::CUDA, true);
    let formatted = format!("{}", x1);
    assert_eq!(formatted, "Tensor<f32>([[1.0 , 2.0 , 3.0 , 4.0 ], [5.0 , 6.0 , 7.0 , 8.0 ], [9.0 , 10.0 , 11.0 , 12.0 ]]\n");
    let x1 = Tensor::<f32>::new(&db, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![4, 3, 1], Device::CUDA, true);
    let formatted = format!("{}", x1);
    assert_eq!(formatted, "Tensor<f32>([[[1.0 ], [2.0 ], [3.0 ]], [[4.0 ], [5.0 ], [6.0 ]], [[7.0 ], [8.0 ], [9.0 ]], [[10.0 ], [11.0 ], [12.0 ]]]\n");
}
