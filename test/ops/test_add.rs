use crate::neuroxide::ops::op::Operation;
use approx::relative_eq;
use neuroxide::{ops::add::Add, types::tensor::Tensor};

#[test]
fn forward() {
    let mut c1c = Tensor::new(vec![15.0], Box::new([1]));
    let mut c2c = Tensor::new(vec![6.0], Box::new([1]));
    let mut result = Add::forward(Box::new([c1c, c2c]));
    assert_eq!(result.lock().unwrap().get_values()[0], 21.0);

    c1c = Tensor::new(vec![15.0, 4.1, 2.3, 34.1], Box::new([2, 2]));
    c2c = Tensor::new(vec![6.0, 3.1, 1.3, 4.1], Box::new([2, 2]));
    result = Add::forward(Box::new([c1c.clone(), c2c.clone()]));
    let result_lock = result.lock().unwrap();
    let res = result_lock.get_values();
    for i in 0..res.len() {
        assert!(relative_eq!(
            res[i],
            c1c.lock().unwrap().get_values()[i] + c2c.lock().unwrap().get_values()[i],
            epsilon = f64::EPSILON
        ));
    }
}

// #[test]
// fn forward_macro() {
//     let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
//     let mut c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
//     let mut c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
//     let mut result = add!(c1c, c2c);
//     assert_eq!(result.data[0], 21.0);
//
//     c1c = Tensor::new(
//         &db,
//         vec![15.0, 4.1, 2.3, 34.1],
//         vec![2, 2],
//         Device::CPU,
//         false,
//     );
//     c2c = Tensor::new(
//         &db,
//         vec![6.0, 3.1, 1.3, 4.1],
//         vec![2, 2],
//         Device::CPU,
//         false,
//     );
//     result = add!(c1c, c2c);
//     for i in 0..result.data.len() {
//         assert!(relative_eq!(
//             result.data[i],
//             c1c.data[i] + c2c.data[i],
//             epsilon = f64::EPSILON
//         ));
//     }
// }
//
// #[cfg(feature = "cuda")]
// #[test]
// fn forward_cuda() {
//     let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
//     let mut c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CUDA, false);
//     let mut c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CUDA, false);
//     let mut result = AddOp::forward(&vec![&c1c, &c2c]);
//     result.cpu();
//     assert_eq!(result.data[0], 21.0);
//
//     c1c = Tensor::<f32>::new(
//         &db,
//         vec![15.0, 4.1, 2.3, 34.1],
//         vec![2, 2],
//         Device::CUDA,
//         false,
//     );
//     c2c = Tensor::<f32>::new(
//         &db,
//         vec![6.0, 3.1, 1.3, 4.1],
//         vec![2, 2],
//         Device::CUDA,
//         false,
//     );
//     result = AddOp::forward(&vec![&c1c, &c2c]);
//     result.cpu();
//     for i in 0..result.data.len() {
//         assert!(relative_eq!(
//             result.data[i],
//             c1c.data[i] + c2c.data[i],
//             epsilon = f32::EPSILON
//         ));
//     }
// }
//
// #[test]
// fn backward() {
//     let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
//     let x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
//     let c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
//     let c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
//     let r1 = MulOp::forward(&vec![&x, &c1c]);
//     let r2 = MulOp::forward(&vec![&x, &c2c]);
//     let mut result = AddOp::forward(&vec![&r1, &r2]);
//     result = MulOp::forward(&vec![&result, &x]);
//     assert!(relative_eq!(result.data[0], 525.0));
//
//     let grad = result.backward(None, Device::CPU);
//     assert!(relative_eq!(grad.get(&x.id).unwrap().data[0], 210.0));
// }
