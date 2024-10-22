use num::NumCast;
use petgraph::prelude::GraphMap;
use crate::ops::op_generic::{Ops, Operation};
use crate::types::device::Device;
use crate::types::tensor::Tensor;
use crate::utils::node_uid::make_node_uid;
use std::fmt::Display;
use std::ops::{Add, Mul};


#[cfg(feature = "cuda")]
extern "C" {
pub fn add_kernel(len: i32, a: *mut f32, b: *mut f32, c: *mut f32) -> CudnnStatusT;
}

pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses


fn get_col<T>(a: &Vec<T>, nrow: usize, ncol: usize) -> Vec<T> 
where T: Copy + Default + Display{
    let mut result = vec![T::default(); nrow];
    for j in 0..ncol {
        for i in (0..a.len()).step_by(ncol) {
            println!("{}", a[i+j]);
        }
        println!();
    }
    result
}

fn get_rows<T>(a: &Vec<T>, nrow: usize, ncol: usize) -> Vec<T> 
where T: Copy + Default + Display{
    let mut result = vec![T::default(); ncol];
    for j in 0..nrow {
        for i in 0..ncol {
            println!("{}", a[i+j*ncol]);
        }
        println!();
    }
    result
}


#[derive(Debug, Clone)]
pub struct MatMulOp;

impl<T> Operation<T> for MatMulOp 
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default + std::fmt::Debug + NumCast + Display{
    fn forward(inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);
        assert!(inputs[0].device == inputs[1].device);
        assert!(inputs[0].dtype.read().unwrap().get_dtype() == inputs[1].dtype.read().unwrap().get_dtype());

        let last_dim_a = inputs[0].shape[inputs[0].shape.len() - 1];        
        let second_last_dim_b = inputs[1].shape[inputs[1].shape.len() - 2];
        println!("last_dim_a: {}, second_last_dim_b: {}", last_dim_a, second_last_dim_b);
        let a_ncols = inputs[0].shape[inputs[0].shape.len() - 1];
        let a_nrows = inputs[0].shape[inputs[0].shape.len() - 2];
        // get_col(&inputs[0].data, a_nrows, a_ncols);
        get_rows(&inputs[0].data, a_nrows, a_ncols);

        // let index = 0;
        // let skip = 2;
        // for i in (0..inputs[0].data.len()).step_by(3) {
        //     let a = inputs[0].data[i];
        //     println!("a: {:?}", a);
        // }
        // let index = 0;
        // let skip = 3;
        // for i in (0..skip) {
        //     let b = inputs[0].data[i];
        //     println!("b: {:?}", b);
        // }

        let t = inputs[0].clone() + inputs[1].clone();
        t
    }

    fn backward(inputs: &Vec<&Tensor<T>>, _grad: Option<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);

        let mut grad_data = vec![T::default(); inputs[0].data.len()];
        if inputs[0].id == inputs[1].id { //c = a + a => dc/da = 2
            todo!();
            // for i in 0..inputs[0].data.len() {
            //     grad_data[i] = T::from(2).unwrap(); 
            // }
            // Tensor {
            //     id: inputs[0].id,
            //     data: grad_data,
            //     shape: inputs[0].shape.clone(),
            //     device: inputs[0].device,
            //     op: Ops::AddEnum,
            //     requires_grad: inputs[0].requires_grad,
            //     op_chain: inputs[0].op_chain.clone(),
            //     op_head: inputs[0].op_head,
            //     dtype: inputs[0].dtype.clone()
            // }
        } else {
            for i in 0..inputs[0].data.len() {
                grad_data[i] = T::from(1).unwrap(); 
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
}

