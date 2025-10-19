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
pub fn mul_kernel(len: i32, a: *mut f32, b: *mut f32, c: *mut*mut f32) -> CudnnStatusT;
}

pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses


#[derive(Debug, Clone)]
pub struct MulOp;

impl<T> Operation<T> for MulOp
where
    T: TensorElement
{
    fn forward(inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);
        assert!(inputs[0].device == inputs[1].device);
        assert!(inputs[0].dtype.read().unwrap().get_dtype() == inputs[1].dtype.read().unwrap().get_dtype());
        let t = inputs[0].clone() * inputs[1].clone();
        // let db = inputs[0].dtype.clone();
        // db.write().unwrap().insert(t.clone());
        // drop(db);
        t
    }

    fn backward(inputs: &Vec<&Tensor<T>>, grad: Option<&Tensor<T>>, _: Device) -> Tensor<T> {
        assert!(inputs.len() == 2);

        //get index of grad in inputs without for loop
        let grad_index = inputs.iter().position(|&x| x.id == grad.unwrap().id).unwrap();


        return inputs[1 - grad_index].clone(); 
    }
}


impl<T> std::ops::Mul for Tensor<T>
where
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + Default + NumCast
{
    type Output = Tensor<T>;

    fn mul(self, other: Tensor<T>) -> Tensor<T> {
        assert!(self.device == other.device);
        
        let mut a = self.clone();
        let mut b = other.clone();
        let mut a_arr = ArrayD::from_shape_vec(a.shape.clone(), a.data.clone()).unwrap();
        let mut b_arr = ArrayD::from_shape_vec(b.shape.clone(), b.data.clone()).unwrap();
        let res = broadcast_shapes_linear(&mut a.shape, &mut b.shape);
        if res.is_err() {
            panic!("Shapes are not broadcastable");
        }
        assert!(a.shape == b.shape);
        a_arr = a_arr.broadcast(a.shape.clone()).unwrap().to_owned();
        b_arr = b_arr.broadcast(b.shape.clone()).unwrap().to_owned();
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
                let res = a_arr * b_arr;
                result = res.iter().map(|&x| x.clone()).collect();
                // result = self.data.iter().zip(other.data.iter()).map(|(a, b)| *a * *b).collect();
            }
            Device::CUDA => {
                #[cfg(feature = "cuda")]
                unsafe {
                    println!("{:?}", self.shape);
                    println!("{:?}", other.shape);
                    println!("{:?}", a.shape);
                    println!("{:?}", b.shape);
                    assert!(self.shape == other.shape);
                    let a_flat = a_arr.as_slice().unwrap();
                    // let b_flat = b_arr.as_slice().unwrap();
                    
                    let len: i32 = a_flat.len() as i32;
                    let mut data: f32 = 0.0;
                    let mut ptr_to_data: *mut f32 = &mut data;
                    mul_kernel(len, a.cuda_ptr.unwrap(), b.cuda_ptr.unwrap(), &mut ptr_to_data);
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
            op: Ops::MulEnum,
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
