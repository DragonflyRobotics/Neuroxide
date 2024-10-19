use num::{Num, NumCast};
use crate::ops::op_generic::{Ops, Operation};
use crate::types::device::Device;
use crate::types::tensor::Tensor;
use crate::utils::node_uid::make_node_uid;
use std::ops::{Add, Mul};

use super::f_to_i_ops::{CosOpTrait, SinOpTrait};


#[cfg(feature = "cuda")]
extern "C" {
pub fn cos_kernel(len: i32, a: *mut f32, c: *mut f32) -> CudnnStatusT;
}

pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses

#[derive(Debug, Clone)]
pub struct CosOp;

impl<T> Operation<T> for CosOp
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default + std::fmt::Debug + NumCast + Num + SinOpTrait + CosOpTrait
{
    fn forward(inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 1);
        let result: Vec<T>; // = vec![T::default(); len as usize];

        match inputs[0].device {
            Device::CPU => {
                result = inputs[0].data.iter().map(|x| x.cos()).collect();
            }
            Device::CUDA => {

                #[cfg(feature = "cuda")]
                unsafe {
                    let a: Vec<f32> = inputs[0].data.iter().map(|&x| <f32 as NumCast>::from(x).unwrap()).collect();
                    let len = a.len() as i32;
                    let mut r = vec![0.0f32; len as usize];
                    cos_kernel(len, a.as_ptr() as *mut f32, r.as_mut_ptr());
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
            op: Ops::CosEnum,
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
            grad_data[i] = T::from(-1).unwrap() * inputs[0].data[i].sin();
        }
        Tensor {
            id: inputs[0].id,
            data: grad_data,
            shape: inputs[0].shape.clone(),
            device: inputs[0].device,
            op: Ops::CosEnum,
            requires_grad: inputs[0].requires_grad,
            op_chain: inputs[0].op_chain.clone(),
            op_head: inputs[0].op_head,
            dtype: inputs[0].dtype.clone()
        }
    }
}
