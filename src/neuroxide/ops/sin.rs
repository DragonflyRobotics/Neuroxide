use num::{Num, NumCast};
use crate::ops::op_generic::{Ops, Operation};
use crate::types::device::Device;
use crate::types::tensor::Tensor;
use crate::utils::node_uid::make_node_uid;
use std::ops::{Add, Mul};

use super::f_to_i_ops::{CosOpTrait, SinOpTrait};


#[cfg(feature = "cuda")]
extern "C" {
pub fn sin_kernel(len: i32, a: *mut f32, c: *mut f32) -> CudnnStatusT;
}

pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses

#[derive(Debug, Clone)]
pub struct SinOp;

impl<T> Operation<T> for SinOp
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default + std::fmt::Debug + NumCast + Num + SinOpTrait + CosOpTrait
{
    fn forward(inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 1);
        let result: Vec<T>; // = vec![T::default(); len as usize];

        match inputs[0].device {
            Device::CPU => {
                result = inputs[0].data.iter().map(|x| x.sin()).collect();
            }
            Device::CUDA => {
                let a: Vec<f32> = inputs[0].data.iter().map(|&x| <f32 as NumCast>::from(x).unwrap()).collect();
                let len = a.len() as i32;

                #[cfg(feature = "cuda")]
                unsafe {
                    let mut r = vec![0.0f32; len as usize];
                    sin_kernel(len, a.as_ptr() as *mut f32, r.as_mut_ptr());
                    result = r.iter().map(|&x| <T as NumCast>::from(x).unwrap()).collect();
                }
                #[cfg(not(feature = "cuda"))]
                {
                    panic!("CUDA feature not enabled");
                }
            }
        }


        //merge graphs
        let mut result_graph = inputs[0].op_chain.clone();

        let result_id = make_node_uid();
        result_graph.add_node(result_id);
        result_graph.add_edge(result_id, inputs[0].op_head, make_node_uid());

        let t = Tensor {
            id: result_id,
            data: result,
            shape: inputs[0].shape.clone(),
            device: inputs[0].device,
            op: Ops::SinEnum,
            requires_grad: inputs[0].requires_grad,
            op_chain: result_graph,
            op_head: result_id,
            dtype: inputs[0].dtype.clone()
        };



        let db = inputs[0].dtype.clone();
        db.write().unwrap().insert(t.clone());
        drop(db);
        t
    }

    fn backward(inputs: &Vec<&Tensor<T>>, _grad: Option<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 1);
        let mut grad_data = vec![T::default(); inputs[0].data.len()];
        for i in 0..inputs[0].data.len() {
            grad_data[i] = inputs[0].data[i].cos();
        }
        Tensor {
            id: inputs[0].id,
            data: grad_data,
            shape: inputs[0].shape.clone(),
            device: inputs[0].device,
            op: Ops::AddEnum,
            requires_grad: inputs[0].requires_grad,
            op_chain: inputs[0].op_chain.clone(),
            op_head: inputs[0].op_head,
            dtype: inputs[0].dtype.clone()
        }
    }
}

//implement add for tensor

// impl<T> std::ops::Add for Tensor<T>
// where
//     T: std::ops::Add<Output = T> + Copy + Default,
// {
//     type Output = Tensor<T>;
//
//     fn add(self, other: Tensor<T>) -> Tensor<T> {
//         assert!(self.shape == other.shape);
//         assert!(self.device == other.device);
//         let result = self.data.iter().zip(other.data.iter()).map(|(a, b)| *a + *b).collect();
//
//         //merge graphs
//         let mut result_graph = GraphMap::new();
//         let self_graph = &self.op_chain;
//         let other_graph = &other.op_chain;
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
//         result_graph.add_edge(result_id, self.op_head, make_node_uid());
//         result_graph.add_edge(result_id, other.op_head, make_node_uid());
//
//         Tensor {
//             id: result_id,
//             data: result,
//             shape: self.shape.clone(),
//             device: self.device,
//             op: Ops::AddEnum,
//             requires_grad: self.requires_grad || other.requires_grad,
//             op_chain: result_graph,
//             op_head: result_id,
//             dtype: self.dtype.clone()
//         }
//     }
// }
