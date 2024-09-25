extern crate test;


use std::sync::{Arc, RwLock};

use neuroxide::types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}};


#[bench]
fn new (b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    b.iter(|| {
        let _ = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    });
    b.iter(|| {
        let _ = Tensor::new(&db, vec![15.0, 2.0, 3.1], vec![3, 1], Device::CPU, false);
    });
    b.iter(|| {
        let _ = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    });
}

#[bench]
fn clear_graph(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let mut x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    x.op_chain.add_node(1);
    x.op_chain.add_node(2);
    x.op_chain.add_edge(1, 2, 1);
    x.op_chain.add_edge(2, 1, 1);
    b.iter(|| {
        x.clear_graph();
    });
}
