use std::sync::{Arc, RwLock};

use neuroxide::{ops::{ln::LnOp, op_generic::Operation, pow::PowOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};


fn main() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let a = Tensor::<f32>::new(&db, vec![0.0, 2.0, 3.0], vec![3], Device::CUDA, true);
    let b = Tensor::<f32>::new(&db, vec![4.0, 5.0, 6.0], vec![3], Device::CUDA, false);

    let c = PowOp::forward(&vec![&a, &b]);
    println!("{}", c);
    let grad = c.backward(None);
    for g in grad {
        println!("{}", g.1);
    }
    
    let d = LnOp::forward(&vec![&a]);
    println!("{}", d);
    let grad = d.backward(None);
    for g in grad {
        println!("{}", g.1);
    }


    let a = Tensor::<f32>::new(&db, vec![1.0, 2.0, 3.0], vec![3], Device::CUDA, false);
    let b = Tensor::<f32>::new(&db, vec![4.0, 5.0, 6.0], vec![3], Device::CUDA, true);

    let c = PowOp::forward(&vec![&a, &b]);
    let grad = c.backward(None);
    for g in grad {
        println!("{}", g.1);
    }
}
