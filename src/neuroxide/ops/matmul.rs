// use ndarray::{array, s, Array2, ArrayD, Axis, Ix2, IxDyn};
// use num::NumCast;
// use petgraph::prelude::GraphMap;
// use crate::ops::op_generic::{Ops, Operation};
// use crate::types::device::Device;
// use crate::types::tensor::Tensor;
// use crate::utils::node_uid::make_node_uid;
// use std::fmt::Display;
// use std::ops::{Add, Mul};
//
//
//
// #[cfg(feature = "cuda")]
// extern "C" {
// pub fn add_kernel(len: i32, a: *mut f32, b: *mut f32, c: *mut f32) -> CudnnStatusT;
// }
//
// pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses
//
//
// #[derive(Debug, Clone)]
// pub struct MatMulOp;
//
// impl<T> Operation<T> for MatMulOp 
// where
//     T: Add<Output = T> + Mul<Output = T> + Copy + Default + std::fmt::Debug + NumCast + Display{
//     fn forward(inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
//         assert!(inputs.len() == 2);
//         assert!(inputs[0].device == inputs[1].device);
//
//         println!("{:?}", inputs[0].shape);
//         println!("{:?}", inputs[1].shape);
//
//         let result: Vec<T>; // = vec![T::default(); len as usize];
//         match inputs[0].device {
//             Device::CPU => {
//                 if inputs[0].shape.len() == 2 && inputs[1].shape.len() == 2 {
//                     let a = Array2::from_shape_vec(Ix2(inputs[0].shape[0], inputs[0].shape[1]), inputs[0].data.clone()).unwrap();
//                     let b = Array2::from_shape_vec(Ix2(inputs[1].shape[0], inputs[1].shape[1]), inputs[1].data.clone()).unwrap();
//                     println!("{:?}", a);
//                     println!("{:?}", b);
//                     // println!("{:?}", a.dot(&b));
//
//                     // let c = a.dot(&b);
//                     // result = c.iter().map(|&x| x).collect();
//                 } else {
//                     panic!("Matrix multiplication only supported for 2D tensors");
//                 }
//             }
//             Device::CUDA => {
//                 #[cfg(feature = "cuda")]
//                 unsafe {
//                     let len: i32 = self.data.len() as i32;
//                     let a: Vec<f32> = self.data.iter().map(|&x| <f32 as NumCast>::from(x).unwrap()).collect();
//                     let b: Vec<f32> = other.data.iter().map(|&x| <f32 as NumCast>::from(x).unwrap()).collect();
//                     let mut r = vec![0.0; len as usize];
//                     add_kernel(len, a.as_ptr() as *mut f32, b.as_ptr() as *mut f32, r.as_mut_ptr());
//                     result = r.iter().map(|&x| <T as NumCast>::from(x).unwrap()).collect();
//                 }
//                 #[cfg(not(feature = "cuda"))]
//                 {
//                     panic!("CUDA feature not enabled");
//                 }
//             }
//         }
//         //merge graphs
//         let mut result_graph = GraphMap::new();
//         let self_graph = &inputs[0].op_chain;
//         let other_graph = &inputs[1].op_chain;
//         let self_nodes = self_graph.nodes();
//         let other_nodes = other_graph.nodes();
//         for node in self_nodes {
//             result_graph.add_node(node);
//         }
//         for node in other_nodes {
//             result_graph.add_node(node);
//         }
//         let self_edges = self_graph.all_edges();
//         let other_edges = other_graph.all_edges();
//         for edge in self_edges {
//             result_graph.add_edge(edge.0, edge.1, make_node_uid());
//         }
//         for edge in other_edges {
//             result_graph.add_edge(edge.0, edge.1, make_node_uid());
//         }
//
//         let result_id = make_node_uid();
//         result_graph.add_node(result_id);
//         result_graph.add_edge(result_id, inputs[0].op_head, make_node_uid());
//         result_graph.add_edge(result_id, inputs[1].op_head, make_node_uid());
//
//         let t = Tensor {
//             id: result_id,
//             data: result,
//             shape: inputs[0].shape.clone(),
//             device: inputs[0].device,
//             op: Ops::MatMulEnum,
//             requires_grad: inputs[0].requires_grad || inputs[1].requires_grad,
//             op_chain: result_graph,
//             op_head: result_id,
//             dtype: inputs[0].dtype.clone()
//         };
//
//         let db = inputs[0].dtype.clone();
//         db.write().unwrap().insert(t.clone());
//         drop(db);
//         t
//     }
//
//     fn backward(inputs: &Vec<&Tensor<T>>, _grad: Option<&Tensor<T>>) -> Tensor<T> {
//         assert!(inputs.len() == 2);
//
//         let mut grad_data = vec![T::default(); inputs[0].data.len()];
//         if inputs[0].id == inputs[1].id { //c = a + a => dc/da = 2
//             todo!();
//             // for i in 0..inputs[0].data.len() {
//             //     grad_data[i] = T::from(2).unwrap(); 
//             // }
//             // Tensor {
//             //     id: inputs[0].id,
//             //     data: grad_data,
//             //     shape: inputs[0].shape.clone(),
//             //     device: inputs[0].device,
//             //     op: Ops::AddEnum,
//             //     requires_grad: inputs[0].requires_grad,
//             //     op_chain: inputs[0].op_chain.clone(),
//             //     op_head: inputs[0].op_head,
//             //     dtype: inputs[0].dtype.clone()
//             // }
//         } else {
//             for i in 0..inputs[0].data.len() {
//                 grad_data[i] = T::from(1).unwrap(); 
//             }
//             Tensor {
//                 id: inputs[0].id,
//                 data: grad_data,
//                 shape: inputs[0].shape.clone(),
//                 device: inputs[0].device,
//                 op: Ops::AddEnum,
//                 requires_grad: inputs[0].requires_grad,
//                 op_chain: inputs[0].op_chain.clone(),
//                 op_head: inputs[0].op_head,
//                 dtype: inputs[0].dtype.clone()
//             }
//         }
//     }
// }
//
