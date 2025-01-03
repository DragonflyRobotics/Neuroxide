use ndarray::ArrayD;
use num::{Num, NumCast};
use petgraph::prelude::GraphMap;
use crate::ops::op_generic::{Ops, Operation};
use crate::types::device::Device;
use crate::types::tensor::Tensor;
use crate::utils::array_utils::broadcast_shapes_linear;
use crate::utils::node_uid::make_node_uid;
use std::ops::{Add, Mul};

use super::f_to_i_ops::{LnOpTrait, PowOpTrait};


#[cfg(feature = "cuda")]
extern "C" {
pub fn pow_kernel(len: i32, a: *mut f32, b: *mut f32, c: *mut f32) -> CudnnStatusT;
}

pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses

#[derive(Debug, Clone)]
pub struct PowOp;

impl<T> Operation<T> for PowOp 
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default + std::fmt::Debug + NumCast + Num + PowOpTrait + LnOpTrait
{
    fn forward(inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);
        assert!(inputs[0].device == inputs[1].device);
        assert!(inputs[0].dtype.read().unwrap().get_dtype() == inputs[1].dtype.read().unwrap().get_dtype());
        
        let mut a = inputs[0].clone();
        let mut b = inputs[1].clone(); 
        let mut a_arr = ArrayD::from_shape_vec(a.shape.clone(), a.data.clone()).unwrap();
        let mut b_arr = ArrayD::from_shape_vec(b.shape.clone(), b.data.clone()).unwrap();
        broadcast_shapes_linear(&mut a.shape, &mut b.shape);
        assert!(a.shape == b.shape);
        a_arr = a_arr.broadcast(a.shape).unwrap().to_owned();
        b_arr = b_arr.broadcast(b.shape).unwrap().to_owned();
        let final_shape: Vec<usize> = a_arr.shape().iter().map(|x| *x as usize).collect();
        
        let result: Vec<T>; // = vec![T::default(); len as usize];

        match inputs[0].device {
            Device::CPU => {
                let a_flat = a_arr.as_slice().unwrap();
                let b_flat = b_arr.as_slice().unwrap();
                result = a_flat.iter().zip(b_flat.iter()).map(|(a, b)| a.pow(*b)).collect();
            }
            Device::CUDA => {
                #[cfg(feature = "cuda")]
                unsafe {
                    let a_flat = a_arr.as_slice().unwrap();
                    let b_flat = b_arr.as_slice().unwrap();
                    
                    let len: i32 = a_flat.len() as i32;
                    let mut r = vec![0.0; len as usize];
                    pow_kernel(len, a_flat.as_ptr() as *mut f32, b_flat.as_ptr() as *mut f32, r.as_mut_ptr());
                    result = r.iter().map(|&x| <T as NumCast>::from(x).unwrap()).collect();
                }
                #[cfg(not(feature = "cuda"))]
                {
                    panic!("CUDA feature not enabled");
                }
            }
        }


            //merge graphs
        let mut result_graph = GraphMap::new();
        let self_graph = &inputs[0].op_chain;
        let other_graph = &inputs[1].op_chain;
        let self_nodes = self_graph.nodes();
        let other_nodes = other_graph.nodes();
        for node in self_nodes {
            result_graph.add_node(node);
        }
        for node in other_nodes {
            result_graph.add_node(node);
        }
        let self_edges = self_graph.all_edges();
        let other_edges = other_graph.all_edges();
        for edge in self_edges {
            result_graph.add_edge(edge.0, edge.1, make_node_uid());
        }
        for edge in other_edges {
            result_graph.add_edge(edge.0, edge.1, make_node_uid());
        }

        let result_id = make_node_uid();
        result_graph.add_node(result_id);
        result_graph.add_edge(result_id, inputs[0].op_head, make_node_uid());
        result_graph.add_edge(result_id, inputs[1].op_head, make_node_uid());

        let t = Tensor {
            id: result_id,
            data: result,
            shape: final_shape,
            device: inputs[1].device,
            op: Ops::PowEnum,
            requires_grad: inputs[0].requires_grad || inputs[1].requires_grad,
            op_chain: result_graph,
            op_head: result_id,
            dtype: inputs[0].dtype.clone()
        };

        let db = inputs[0].dtype.clone();
        db.write().unwrap().insert(t.clone());
        drop(db);
        t

    }

    fn backward(inputs: &Vec<&Tensor<T>>, grad: Option<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);
        // println!("Pow got {} ^ {} ", inputs[0], inputs[1]);
        
        let mut a = inputs[0].clone();
        let mut b = inputs[1].clone();
        let mut a_arr = ArrayD::from_shape_vec(a.shape.clone(), a.data.clone()).unwrap();
        let mut b_arr = ArrayD::from_shape_vec(b.shape.clone(), b.data.clone()).unwrap();
        broadcast_shapes_linear(&mut a.shape, &mut b.shape);
        assert!(a.shape == b.shape);
        a_arr = a_arr.broadcast(a.shape).unwrap().to_owned();
        b_arr = b_arr.broadcast(b.shape).unwrap().to_owned();
        let a_flat = a_arr.as_slice().unwrap();
        let b_flat = b_arr.as_slice().unwrap();
        let final_shape: Vec<usize> = a_arr.shape().iter().map(|x| *x as usize).collect();
        // println!("FINAL SHAPE {:?}", final_shape);
        
        let mut grad_data = vec![T::default(); inputs[0].data.len()];
        let dx_index = if grad.unwrap().id == inputs[0].id {0} else {1};
        for i in 0..inputs[0].data.len() {
            if dx_index == 0 {
                grad_data[i] = b_flat[i] * a_flat[i].pow(b_flat[i] - T::one());
            } else {
                grad_data[i] = a_flat[i].ln() * a_flat[i].pow(b_flat[i]);
            }
        }
        // println!("GRAD DATA {:?}", grad_data);
        Tensor {
            id: inputs[0].id,
            data: grad_data,
            shape: final_shape,
            device: inputs[0].device,
            op: Ops::PowEnum,
            requires_grad: inputs[0].requires_grad,
            op_chain: inputs[0].op_chain.clone(),
            op_head: inputs[0].op_head,
            dtype: inputs[0].dtype.clone()
        }
    }
}
