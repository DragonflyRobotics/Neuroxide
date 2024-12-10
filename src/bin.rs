use std::sync::{Arc, RwLock};

use ndarray::{array, linalg::general_mat_mul, Array2, ArrayD, Ix2, IxDyn};
use neuroxide::{types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use neuroxide::ops::op_generic::Operation;

#[macro_use]
extern crate neuroxide;


fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::new(&db, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], Device::CPU, true);
    let x2 = Tensor::new(&db, vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![3, 2], Device::CPU, true);

    println!("{}", x);
    println!("{}", x2);
    // let a = Array2::from_shape_vec([x.shape[0], x.shape[1]], x.data.clone()).unwrap();
    // let b = Array2::from_shape_vec([x2.shape[0], x2.shape[1]], x2.data.clone()).unwrap();
    let a = ArrayD::from_shape_vec(IxDyn(&x.shape), x.data.clone()).unwrap();
    let b = ArrayD::from_shape_vec(IxDyn(&x2.shape), x2.data.clone()).unwrap();
    let a = a.into_dimensionality::<Ix2>().unwrap();
    let b = b.into_dimensionality::<Ix2>().unwrap();
    println!("{:?}", a);
    println!("{:?}", b);
    println!("{:?}", a.dot(&b));
    // let c = MatMulOp::forward(&vec![&x, &x2]);


    // Perform matrix multiplication using `general_mat_mul`

    // println!("{:?}", a.dot(&b));
}
