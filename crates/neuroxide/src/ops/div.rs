use ndarray::ArrayD;
use num::NumCast;
use petgraph::prelude::GraphMap;
use crate::ops::op_generic::{Ops, Operation};
use crate::types::device::Device;
use crate::types::tensor::Tensor;
use crate::types::t::TensorElement;
use crate::utils::array_utils::broadcast_shapes_linear;
use crate::utils::node_uid::make_node_uid;
use cfg_if::cfg_if;

#[cfg(feature = "cuda")]
unsafe extern "C" {
pub fn div_kernel(len: i32, a: *mut f32, b: *mut f32, c: *mut*mut f32) -> CudnnStatusT;
}

pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses


#[derive(Debug, Clone)]
pub struct DivOp;

impl<T> Operation<T> for DivOp
where
    T: TensorElement
{
    fn forward(inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);
        assert!(inputs[0].device == inputs[1].device);
        assert!(inputs[0].dtype.read().unwrap().get_dtype() == inputs[1].dtype.read().unwrap().get_dtype());
        let t = inputs[0].clone() / inputs[1].clone();
        // let db = inputs[0].dtype.clone();
        // db.write().unwrap().insert(t.clone());
        // drop(db);
        t
    }

    fn backward(inputs: &Vec<&Tensor<T>>, grad: Option<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);

        let f = inputs[0].clone();         
        let g = inputs[1].clone();
        let dx_index = if grad.unwrap().id == inputs[0].id {0} else {1};
        let mut grad_data = vec![T::default(); inputs[0].data.len()];

        if dx_index == 0 {
            for i in 0..inputs[0].data.len() {
                grad_data[i] = T::one() / g.data[i];
            }
        } else {
            for i in 0..inputs[0].data.len() {
                grad_data[i] = T::from(-1.0).unwrap() * (f.data[i] / (g.data[i] * g.data[i]));
            }
        }
        // println!("Div Grad Data {:?}", grad_data);
        // println!("Div Shape {:?}", inputs[0].shape);

        Tensor {
            id: inputs[0].id,
            data: grad_data,
            shape: inputs[0].shape.clone(),
            device: Device::CPU,
            op: Ops::DivEnum,
            requires_grad: inputs[0].requires_grad,
            op_chain: inputs[0].op_chain.clone(),
            op_head: inputs[0].op_head,
            dtype: inputs[0].dtype.clone(),
            cuda_ptr: None // TODO: Fix this
        }
    }
}


impl<T> std::ops::Div for Tensor<T>
where
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + std::ops::Div<Output = T> + Copy + Default + NumCast
{
    type Output = Tensor<T>;

    fn div(self, other: Tensor<T>) -> Tensor<T> {
        assert!(self.device == other.device);

        let mut a = self.clone();
        let mut b = other.clone();
        let mut a_arr = ArrayD::from_shape_vec(a.shape.clone(), a.data.clone()).unwrap();
        // let mut b_arr = ArrayD::from_shape_vec(b.shape.clone(), b.data.clone()).unwrap();
        let res = broadcast_shapes_linear(&mut a.shape, &mut b.shape);
        res.unwrap();
        assert!(a.shape == b.shape);
        a_arr = a_arr.broadcast(a.shape).unwrap().to_owned();
        // b_arr = b_arr.broadcast(b.shape).unwrap().to_owned();

        let final_shape: Vec<usize> = a_arr.shape().iter().map(|x| *x as usize).collect();
        let result: Vec<T>;//vec![T::default(); len as usize];
        cfg_if! {
            if #[cfg(feature = "cuda")] {
                let mut cuda_ptr: Option<*mut f32> = None;
            } else {
                let cuda_ptr: Option<*mut f32> = None;
            }
        }
        match self.device {
            Device::CPU => {
                result = self.data.iter().zip(other.data.iter()).map(|(a, b)| *a / *b).collect();
            }
            Device::CUDA => {
                #[cfg(feature = "cuda")]
                unsafe {
                    assert!(self.shape == other.shape);
                    let a_flat = a_arr.as_slice().unwrap();
                    // let b_flat = b_arr.as_slice().unwrap();
                    
                    let len: i32 = a_flat.len() as i32;
                    let mut data: f32 = 0.0;
                    let mut ptr_to_data: *mut f32 = &mut data;
                    div_kernel(len, a.cuda_ptr.unwrap(), b.cuda_ptr.unwrap(), &mut ptr_to_data);
                    cuda_ptr = Some(ptr_to_data);
                    let r = vec![0.0; len as usize];
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
        let self_graph = &self.op_chain;
        let other_graph = &other.op_chain;
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
        result_graph.add_edge(result_id, self.op_head, make_node_uid());
        result_graph.add_edge(result_id, other.op_head, make_node_uid());

        let t = Tensor {
            id: result_id,
            data: result,
            shape: final_shape,
            device: self.device,
            op: Ops::DivEnum,
            requires_grad: self.requires_grad || other.requires_grad,
            op_chain: result_graph,
            op_head: result_id,
            dtype: self.dtype.clone(),
            cuda_ptr
        };

        let db = self.dtype.clone();
        db.write().unwrap().insert(t.clone());
        drop(db);
        t
    }
}

