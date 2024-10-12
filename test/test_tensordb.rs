use std::sync::{Arc, RwLock};

use neuroxide::{ops::op_generic::Ops, types::{device::Device, tensor::Tensor, tensordb::{self, DTypes, TensorDB}}};

#[test]
fn new() {
    let db = Arc::new(RwLock::new(TensorDB::<f32>::new(DTypes::F32)));
    assert!(db.read().unwrap().get_dtype() == DTypes::F32);
}

#[test]
fn insert_get() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let x = Tensor {
        id: 0,
        data: vec![5.0],
        shape: vec![1],
        device: Device::CPU,
        requires_grad: true,
        op_chain: Default::default(),
        op_head: 0,
        op: Ops::TensorEnum,
        dtype: db.clone()
    };
    db.write().unwrap().insert(x.clone());
    assert!(db.read().unwrap().get(0).unwrap().data == vec![5.0]);
}

#[test]
fn get_dtype() {
    let db = Arc::new(RwLock::new(TensorDB::<f32>::new(DTypes::F32)));
    assert!(db.read().unwrap().get_dtype() == DTypes::F32);
}


// pub enum DTypes {
//     I8,
//     I16,
//     I32,
//     I64,
//     I128,
//     U8,
//     U16,
//     U32,
//     U64,
//     U128,
//     F32,
//     F64,
//     Bool,
//     Char
// }

#[test]
fn type_check() {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I8)));
    let c1c = Tensor::new(&db, vec![15_i8], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::I8, c1c.data[0]);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I16)));
    let c1c = Tensor::new(&db, vec![15_i16], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::I16, c1c.data[0]);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I32)));
    let c1c = Tensor::new(&db, vec![15_i32], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::I32, c1c.data[0]);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I64)));
    let c1c = Tensor::new(&db, vec![15_i64], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::I64, c1c.data[0]);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::I128)));
    let c1c = Tensor::new(&db, vec![15_i128], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::I128, c1c.data[0]);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::U8)));
    let c1c = Tensor::new(&db, vec![15_u8], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::U8, c1c.data[0]);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::U16)));
    let c1c = Tensor::new(&db, vec![15_u16], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::U16, c1c.data[0]);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::U32)));
    let c1c = Tensor::new(&db, vec![15_u32], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::U32, c1c.data[0]);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::U64)));
    let c1c = Tensor::new(&db, vec![15_u64], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::U64, c1c.data[0]);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::U128)));
    let c1c = Tensor::new(&db, vec![15_u128], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::U128, c1c.data[0]);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let c1c = Tensor::new(&db, vec![15.0_f32], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::F32, c1c.data[0]);

    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
    tensordb::assert_types(DTypes::F64, c1c.data[0]);

    // let db = Arc::new(RwLock::new(TensorDB::new(DTypes::Bool)));
    // let c1c = Tensor::new(&db, vec![true], vec![1], Device::CPU, false);
    // tensordb::assert_types(DTypes::Bool, c1c.data[0]);
    //
    // let db = Arc::new(RwLock::new(TensorDB::new(DTypes::Char)));
    // let c1c = Tensor::new(&db, vec!['a'], vec![1], Device::CPU, false);
    // tensordb::assert_types(DTypes::Char, c1c.data[0]);
}
