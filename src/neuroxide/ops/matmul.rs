use ndarray::{Array1, Array2, Array3, ArrayD, Axis, Ix1, Ix2, Ix3, IxDyn};
use num::NumCast;
use petgraph::prelude::GraphMap;
use crate::ops::op_generic::{Ops, Operation};
use crate::types::device::Device;
use crate::types::tensor::Tensor;
use crate::utils::array_utils::broadcast_shapes_matmul;
use crate::utils::node_uid::make_node_uid;
use std::ops::{Add, Mul};


#[cfg(feature = "cuda")]
extern "C" {
pub fn matmul(m: i32, n: i32, k: i32, h_A: *mut f32, h_B: *mut f32, h_C: *mut f32) -> CudnnStatusT;
}

pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses
//
#[derive(Debug, Clone)]
pub struct MatMulOp;


impl<T> Operation<T> for MatMulOp 
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default + std::fmt::Debug + Clone + NumCast + ndarray::ScalarOperand + ndarray::LinalgScalar //+ Not<Output = T>
{
    fn forward(inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);
        assert!(inputs[0].device == inputs[1].device);

        let mut shape1 = inputs[0].shape.clone();
        let mut shape2 = inputs[1].shape.clone();

        let shape = broadcast_shapes_matmul(&mut shape1, &mut shape2);
        println!("Broadcasted shape {:?} to shape {:?}", shape1, shape2);

        let result: Vec<T>; // = vec![T::default(); len as usize];
        match inputs[0].device {
            Device::CPU => {
                if shape1.len() == 1 && shape2.len() == 1 {
                    let a = ArrayD::<T>::from_shape_vec(IxDyn(&shape1), inputs[0].data.clone()).unwrap();
                    let b = ArrayD::<T>::from_shape_vec(IxDyn(&shape2), inputs[1].data.clone()).unwrap();
                    let a: Array1<T> = a.into_dimensionality::<Ix1>().unwrap();
                    let b: Array1<T> = b.into_dimensionality::<Ix1>().unwrap();


                    let c = a.dot(&b);
                    result = vec![c];
                    // shape = vec![1];
                    // result = c.iter().map(|&x| x).collect();
                    // shape = vec![c.shape()[0], c.shape()[1]];
                }
                else if shape1.len() == 2 && shape2.len() == 2 {
                    let a = ArrayD::<T>::from_shape_vec(IxDyn(&shape1), inputs[0].data.clone()).unwrap();
                    let b = ArrayD::<T>::from_shape_vec(IxDyn(&shape2), inputs[1].data.clone()).unwrap();
                    let a: Array2<T> = a.into_dimensionality::<Ix2>().unwrap();
                    let b: Array2<T> = b.into_dimensionality::<Ix2>().unwrap();
                    let c = a.dot(&b);
                    result = c.iter().map(|&x| x).collect();
                    // shape = vec![c.shape()[0], c.shape()[1]];
                } 
                else if shape1.len() == 3 && shape2.len() == 3 {
                    let a = ArrayD::<T>::from_shape_vec(IxDyn(&shape1), inputs[0].data.clone()).unwrap();
                    let b = ArrayD::<T>::from_shape_vec(IxDyn(&shape2), inputs[1].data.clone()).unwrap();
                    let a: Array3<T> = a.into_dimensionality::<Ix3>().unwrap();
                    let b: Array3<T> = b.into_dimensionality::<Ix3>().unwrap();

                    assert!(a.shape()[0] == b.shape()[0]); //batch size
                    assert!(a.shape()[2] == b.shape()[1]); //inner dimension

                    let batch_size = a.shape()[0];
                    let m = a.shape()[1];
                    let n = b.shape()[2];

                    // Initialize the output array
                    let mut c = Array3::<T>::zeros((batch_size, m, n));
                    // Perform batched matrix multiplication
                    for i in 0..batch_size {
                        let a_slice: Array2<T> = a.index_axis(Axis(0), i).to_owned(); // (m, k)
                        let b_slice: Array2<T> = b.index_axis(Axis(0), i).to_owned(); // (k, n)
                        c.index_axis_mut(Axis(0), i).assign(&a_slice.dot(&b_slice));
                    }
                    result = c.iter().map(|&x| x).collect();
                    // shape = vec![c.shape()[0], c.shape()[1], c.shape()[2]];
                }
                else if shape1.len() == 4 && shape2.len() == 4 {
                    todo!();
                }
                else {
                    panic!("Matrix multiplication only supported for 2D tensors");
                }
            }
            Device::CUDA => {
                if shape1.len() == 1 && shape2.len() == 1 {
                    let a = ArrayD::<T>::from_shape_vec(IxDyn(&shape1), inputs[0].data.clone()).unwrap();
                    let b = ArrayD::<T>::from_shape_vec(IxDyn(&shape2), inputs[1].data.clone()).unwrap();
                    let a: Array1<T> = a.into_dimensionality::<Ix1>().unwrap();
                    let b: Array1<T> = b.into_dimensionality::<Ix1>().unwrap();

                    let a_vec: Vec<T> = a.iter().map(|&x| x).collect();
                    let b_vec: Vec<T> = b.iter().map(|&x| x).collect();
                    let a_shape = a.shape().to_vec();
                    // shape = vec![1];
                    result = vec![T::default(); shape.iter().product()];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        matmul(1 as i32, 1 as i32, a_shape[0] as i32, a_vec.as_ptr() as *mut f32, b_vec.as_ptr() as *mut f32, result.as_ptr() as *mut f32); 
                    }
                }
                else if shape1.len() == 2 && shape2.len() == 2 {
                    let a_shape = shape1.clone();
                    let b_shape = shape2.clone();
                    let a = inputs[0].data.clone();
                    let b = inputs[1].data.clone();

                    // shape = vec![a_shape[0], b_shape[1]];
                    result = vec![T::default(); shape.iter().product()];
                    #[cfg(feature = "cuda")]
                    unsafe {
                        matmul(a_shape[0] as i32, b_shape[1] as i32, a_shape[1] as i32, a.as_ptr() as *mut f32, b.as_ptr() as *mut f32, result.as_ptr() as *mut f32); 
                    }
                } 
                else if shape1.len() == 3 && shape2.len() == 3 {
                    let a = ArrayD::<T>::from_shape_vec(IxDyn(&shape1), inputs[0].data.clone()).unwrap();
                    let b = ArrayD::<T>::from_shape_vec(IxDyn(&shape2), inputs[1].data.clone()).unwrap();
                    let a: Array3<T> = a.into_dimensionality::<Ix3>().unwrap();
                    let b: Array3<T> = b.into_dimensionality::<Ix3>().unwrap();

                    assert!(a.shape()[0] == b.shape()[0]); //batch size
                    assert!(a.shape()[2] == b.shape()[1]); //inner dimension

                    let batch_size = a.shape()[0];
                    let m = a.shape()[1];
                    let n = b.shape()[2];

                    // Initialize the output array
                    let mut c = Array3::<T>::zeros((batch_size, m, n));
                    // Perform batched matrix multiplication
                    for i in 0..batch_size {
                        let a_slice: Array2<T> = a.index_axis(Axis(0), i).to_owned(); // (m, k)
                        let b_slice: Array2<T> = b.index_axis(Axis(0), i).to_owned(); // (k, n)
                        let a_vec: Vec<T> = a_slice.iter().map(|&x| x).collect();
                        let b_vec: Vec<T> = b_slice.iter().map(|&x| x).collect();
                        let mut c_vec = vec![T::default(); m * n];
                        #[cfg(feature = "cuda")]
                        unsafe {
                            matmul(m as i32, n as i32, a_slice.shape()[1] as i32, a_vec.as_ptr() as *mut f32, b_vec.as_ptr() as *mut f32, c_vec.as_ptr() as *mut f32); 
                        }
                        let c_slice = Array2::<T>::from_shape_vec(Ix2(m, n), c_vec).unwrap();
                        c.index_axis_mut(Axis(0), i).assign(&c_slice);
                    }
                    result = c.iter().map(|&x| x).collect();
                    // shape = vec![c.shape()[0], c.shape()[1], c.shape()[2]];
                }
                else if shape1.len() == 4 && shape2.len() == 4 {
                    todo!();
                }
                else {
                    panic!("Matrix multiplication only supported for 2D tensors");
                }
            }
        }
        println!("FINAL SHAPE: {:?}", shape);
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
            shape, 
            device: inputs[0].device,
            op: Ops::MatMulEnum,
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

        //get index of grad in inputs without for loop
        let grad_index = inputs.iter().position(|&x| x.id == grad.unwrap().id).unwrap();

        let b = inputs[1 - grad_index].data.clone(); 
        let b_shape = inputs[1 - grad_index].shape.clone();

        let b_t: ArrayD<T>;
        let b_t_shape: Vec<usize>;

        if b_shape.len() == 1 {
            b_t = ArrayD::<T>::from_shape_vec(IxDyn(&b_shape), b).unwrap().to_owned();
            b_t_shape = b_t.shape().to_vec();
        }
        else if b_shape.len() == 2 {
            b_t = ArrayD::<T>::from_shape_vec(IxDyn(&b_shape), b).unwrap().t().to_owned();
            b_t_shape = b_t.shape().to_vec();
        } else if b_shape.len() == 3 {
            b_t = ArrayD::<T>::from_shape_vec(IxDyn(&b_shape), b).unwrap().permuted_axes(IxDyn(&[0, 2, 1])).to_owned();
            b_t_shape = b_t.shape().to_vec();
        } else if b_shape.len() == 4 {
            todo!();
        } else {
            panic!("Matrix multiplication grad only supported for <=3D tensors");
        }



        Tensor {
            id: inputs[1 - grad_index].id,
            data: b_t.iter().map(|&x| x).collect(),
            shape: b_t_shape,
            device: inputs[1 - grad_index].device,
            op: Ops::MatMulEnum,
            requires_grad: inputs[1 - grad_index].requires_grad,
            op_chain: inputs[1 - grad_index].op_chain.clone(),
            op_head: inputs[1 - grad_index].op_head,
            dtype: inputs[1 - grad_index].dtype.clone()
        }
    }
}

