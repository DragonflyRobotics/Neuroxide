#![allow(warnings)]
use neuroxide::ops::add::AddOp;
use neuroxide::ops::mul::MulOp;
use neuroxide::ops::op_generic::Operation;
use neuroxide::types::tensor::Tensor;
use neuroxide::types::device::Device;
use neuroxide::types::tensordb::TensorDB;
use petgraph::dot::Dot;

fn main() {
    let mut db = TensorDB::new();// TENSOR_DB.lock().unwrap();
    let x = Tensor::new(&mut db, vec![5.0], vec![1], Device::CPU, true);
    let c1c = Tensor::new(&mut db, vec![15.0], vec![1], Device::CPU, false);
    let c2c = Tensor::new(&mut db, vec![6.0], vec![1], Device::CPU, false);
    let r1 = MulOp::forward(&MulOp, &mut db, &vec![&x, &c1c]);
    let r2 = MulOp::forward(&MulOp, &mut db, &vec![&x, &c2c]);
    let mut result = AddOp::forward(&AddOp, &mut db, &vec![&r1, &r2]);
    result = MulOp::forward(&MulOp, &mut db, &vec![&result, &x]);
    println!("{:?}", result);

    let grad = result.backward(&mut db, None);
    for g in grad.keys() {
        println!("{}", grad.get(g).unwrap().data[0]);
    }


}
