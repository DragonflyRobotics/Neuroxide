use std::sync::{Arc, RwLock};

use neuroxide::types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}};


#[test]
fn new () {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    assert!(x.data[0] == 5.0);
    assert!(x.shape[0] == 1);
    assert!(x.device == Device::CPU);
    assert!(x.requires_grad == true);

    let c1c = Tensor::new(&db, vec![15.0, 2.0, 3.1], vec![3, 1], Device::CPU, false);
    assert!(c1c.data[0] == 15.0);
    assert!(c1c.data[1] == 2.0);
    assert!(c1c.data[2] == 3.1);
    assert!(c1c.shape[0] == 3);
    assert!(c1c.shape[1] == 1);
    assert!(c1c.device == Device::CPU);
    assert!(c1c.requires_grad == false);
}

#[test]
fn clear_graph() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let mut x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    x.op_chain.add_node(1);
    x.op_chain.add_node(2);
    x.op_chain.add_edge(1, 2, 1);
    x.op_chain.add_edge(2, 1, 1);
    assert!(x.op_chain.node_count() == 2+1);
    assert!(x.op_chain.edge_count() == 2);
    x.clear_graph();
    assert!(x.op_chain.node_count() == 1);
    assert!(x.op_chain.edge_count() == 0);
}
    
#[test]
fn dtypes() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    // check that the value of 5.0 is of type f32
    assert!(x.data[0] == 5.0);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    // check that the value of 5.0 is of type f64
    assert!(x.data[0] == 5.0);


    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let x = Tensor::new(&db, vec![5], vec![1], Device::CPU, true);
    // check that the value of 5 is of type i32
    assert!(x.data[0] == 5);
}
