use std::sync::{Arc, RwLock};
use neuroxide::ops::op_generic::Operation;
use neuroxide::{ops::matmul::MatMulOp, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};

fn main() {
    let mut db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let a = Tensor::<f32>::new(
        &db,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
        vec![2, 1, 2, 3],
        Device::CPU,
        true,
    );
    let b = Tensor::<f32>::new(
        &db,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0],
        vec![2, 3, 3, 2],
        Device::CPU,
        true,
    );

    let mut c = MatMulOp::forward(&vec![&a, &b]);
    let grad = c.backward(None);
    
    println!("Grad: {}", grad.get(&a.id).unwrap());
    println!("Grad: {}", grad.get(&b.id).unwrap());
    // println!("Result: {:?}", c.shape);
    // println!("Result: {}", c);
    //
    // let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    // let x = Tensor::new(&db,vec![1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6], vec![2, 2, 3], Device::CPU, true);
    // let x2 = Tensor::new(&db, vec![5, 6, 7, 8, 9, 10], vec![1, 3, 2], Device::CPU, true);
    //
    // let c = MatMulOp::forward(&vec![&x, &x2]);
    // let grad = c.backward(Some(vec![x.id, x2.id]));
    // println!("Grad x: {}", grad.get(&x.id).unwrap());
    // println!("Grad x2: {}", grad.get(&x2.id).unwrap());
}
