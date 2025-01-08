use std::{collections::HashMap, sync::{Arc, RwLock}};

use crate::{ops::{add::AddOp, cos::CosOp, div::DivOp, f_to_i_ops::{CosOpTrait, LnOpTrait, PowOpTrait, SinOpTrait}, ln::LnOp, matmul::MatMulOp, mul::MulOp, op_generic::{Operation, Ops}, pow::PowOp, sin::SinOp, sub::SubOp}, types::{device::Device, tensordb::DTypes}, utils::types::print_type_of};
use ndarray::{ArrayD, IxDyn};
use num::{Num, NumCast};
use petgraph::{algo, prelude::GraphMap, Directed, Direction::Outgoing};
use crate::utils::node_uid::make_node_uid;
use rand::Rng;

use super::tensordb::{assert_types, TensorDB};

#[derive(Clone)]
pub struct Tensor<T, const N: usize, const M: usize> {
    pub id: i32,
    pub data: [T; N],
    pub shape: [usize; M],
    pub device: Device,
    pub op: Ops,
    pub requires_grad: bool,
    pub op_chain: GraphMap<i32, i32, Directed>,
    pub op_head: i32,
    pub dtype: Arc<RwLock<TensorDB<T>>>
}

impl<T, const N: usize, const M: usize> Tensor<T, N, M> 
where
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + Default + std::fmt::Debug + NumCast + SinOpTrait + CosOpTrait + PowOpTrait + LnOpTrait + Num + ndarray::ScalarOperand + ndarray::LinalgScalar
{
    pub fn new(db: &Arc<RwLock<TensorDB<T>>>, data: [T; N], shape: [usize; M], device: Device, requires_grad: bool) -> Tensor<T, N, M> {
        assert_types(db.read().unwrap().get_dtype(), data[0]);
        if device == Device::CUDA {
            assert!(db.read().unwrap().get_dtype() == DTypes::F32, "CUDA only supports f32");
        }
        let mut graph = GraphMap::new();
        let id = make_node_uid();
        graph.add_node(id);
        let t = Tensor {
            id,
            data,
            shape,
            device,
            op: Ops::TensorEnum,
            requires_grad,
            op_chain: graph,
            op_head: id,
            dtype: db.clone()
        };
        db.write().unwrap().insert(t.clone());
        t
    }

    // pub fn new_ones(db: &Arc<RwLock<TensorDB<T>>>, shape: [usize; M], device: Device, requires_grad: bool) -> Tensor<T, N, M> {
    //     let size = shape.iter().product();
    //     let data: Box<[T]> = vec![T::from(1).unwrap(); size].into_boxed_slice();
    //     let data: [T; N] = [T::from(1).unwrap(); shape.iter().product()]; 
    //     Tensor::new(db, data, shape, device, requires_grad)
    // }
    //
    // pub fn new_zeros(db: &Arc<RwLock<TensorDB<T>>>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor<T> {
    //     let data = vec![T::from(0).unwrap(); shape.iter().product()];
    //     Tensor::new(db, data, shape, device, requires_grad)
    // }
    //
    // pub fn new_uniform(db: &Arc<RwLock<TensorDB<T>>>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor<T> {
    //     let mut rng = rand::thread_rng();
    //     let data = (0..shape.iter().product()).map(|_| T::from(rng.gen::<f32>()).unwrap()).collect(); 
    //     Tensor::new(db, data, shape, device, requires_grad)
    // }

    fn match_ops(&self, d: &Tensor<T>, dx: &Tensor<T>, inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        match d.op {
            Ops::AddEnum => {
                AddOp::backward(inputs, Some(dx))
            },
            Ops::SubEnum => {
                SubOp::backward(inputs, Some(dx))
            },
            Ops::MulEnum => {
                MulOp::backward(inputs, Some(dx))
            },
            Ops::SinEnum => {
                SinOp::backward(inputs, Some(dx))
            },
            Ops::CosEnum => {
                CosOp::backward(inputs, Some(dx))
            },
            Ops::PowEnum => {
                PowOp::backward(inputs, Some(dx))
            },
            Ops::LnEnum => {
                LnOp::backward(inputs, Some(dx))
            },
            Ops::DivEnum => {
                DivOp::backward(inputs, Some(dx))
            },
            Ops::MatMulEnum => {
                MatMulOp::backward(inputs, Some(dx))
            },
            _ => panic!("Operation not implemented")
        }
    }

    pub fn backward(&self, dx: Option<Vec<i32>>) -> HashMap<i32, Tensor<T>> {
        let mut all_leaves = Vec::new();
        {
            let db = self.dtype.read().unwrap();
            match dx {
                Some(x) => {
                    // println!("dx: {:?}", x);
                    for node in x {
                        all_leaves.push(node);
                    }
                },
                None => {
                    for node in self.op_chain.nodes() {
                        let outgoing_edges = self.op_chain.edges_directed(node, Outgoing);
                        if outgoing_edges.count() == 0 && db.get(node).unwrap().requires_grad {
                            all_leaves.push(node);
                        }
                    }
                }
            }
        }
        let mut paths = HashMap::new(); 
        for leaf in all_leaves.clone() {
            let path = algo::all_simple_paths::<Vec<_>, _>(&self.op_chain, self.id, leaf, 0, None).collect::<Vec<_>>();
            paths.insert(leaf, path);
        }
        // println!("All paths: {:?}", paths);

        let mut grad = HashMap::new();

        for leaf in all_leaves.clone() {
            let data = vec![T::from(1).unwrap(); self.data.len()];
            let mut new_graph = GraphMap::<i32, i32, Directed>::new();
            new_graph.add_node(self.id);
            let new_tensor = Tensor {
                id: self.id,
                data,
                shape: self.shape.clone(),
                device: self.device.clone(),
                op: self.op.clone(),
                requires_grad: self.requires_grad,
                op_chain: new_graph,
                op_head: self.id,
                dtype: self.dtype.clone()
            };
            grad.insert(leaf, new_tensor);
        }

        for leaf in all_leaves.clone() {
            let path = &paths[&leaf];
            let mut arr: Vec<Tensor<T>> = Vec::new();
            for p in path {
                // println!("Path: {:?}", p);
                let mut temp = grad[&leaf].clone();
                grad.get_mut(&leaf).unwrap().op_head = grad.get(&leaf).unwrap().id;
                for i in 0..p.len() - 1 {
                    // println!("d{:?}/d{:?}", p[i], p[i + 1]);
                    let neighbor = self.op_chain.neighbors_directed(p[i], Outgoing).collect::<Vec<_>>();
                    // println!("{:?} = {:?} + {:?}", p[i], neighbor[0], neighbor[1]);
                    let mut inputs = vec![];
                    {
                        let db = self.dtype.read().unwrap();
                        for n in neighbor {
                            // println!("n: {:?}", n);
                            inputs.push(db.get(n).unwrap());
                            // println!("inputs: ");
                        }
                        // let inputs = vec![db.get(neighbor[0]).unwrap(), db.get(neighbor[1]).unwrap()];
                        let op_type = db.get(p[i]).unwrap().op.clone();
                        let input_shapes = inputs.iter().map(|x| x.shape.len()).collect::<Vec<_>>();
                        let output = self.match_ops(db.get(p[i]).unwrap(), db.get(p[i+1]).unwrap(), &inputs);
                        // println!("output: {}", output);
                        let grad_index = inputs.iter().position(|&x| x.id == db.get(p[i+1]).unwrap().id).unwrap();
                        drop(db);
                        if let Ops::MatMulEnum = op_type {
                           // output is b_t and temp is downstream so follow upstream dot b_t
                           if input_shapes[0] > 1 || input_shapes[1] > 1 {
                               // println!("temp: {}", temp);
                               // println!("output: {}", output);
                               if grad_index == 0 {
                                   temp = MatMulOp::forward(&vec![&temp, &output]);
                               } else {
                                   temp = MatMulOp::forward(&vec![&output, &temp]);
                               }
                           } else {
                               temp = MulOp::forward(&vec![&output, &temp]);
                           }
                        } else {
                            temp = MulOp::forward(&vec![&output, &temp]);
                        }
                    }
                    // println!("output: ");
                    // println!("grad: ");
                    grad.get_mut(&leaf).unwrap().op_chain.add_edge(p[i], p[i + 1], 0);
                    // println!("grad: ");
                }
                arr.push(temp);
            }
            let mut sum = arr[0].clone();
            for i in 1..arr.len() {
                sum = AddOp::forward(&vec![&sum, &arr[i].clone()]);
            }
            grad.get_mut(&leaf).unwrap().data = sum.data;
            grad.get_mut(&leaf).unwrap().shape = sum.shape;
        }
        grad
    }

    pub fn clear_graph(&mut self) {
        self.op_chain = GraphMap::new();
        self.op_chain.add_node(self.id);
        self.op_head = self.id;
    }
    
    pub fn t(&self) -> Tensor<T> { // TODO: Should I change OpChain?
        let b_arr = ArrayD::from_shape_vec(IxDyn(&self.shape), self.data.clone()).unwrap();
        let b_shape = b_arr.shape();
        let b_t: ArrayD<T>;
        let b_t_shape: Vec<usize>;

        if b_shape.len() == 1 {
            b_t = b_arr; 
            b_t_shape = b_t.shape().to_vec();
        }
        else if b_shape.len() == 2 {
            b_t = b_arr.t().to_owned();
            b_t_shape = b_t.shape().to_vec();
        } else if b_shape.len() == 3 {
            b_t = b_arr.permuted_axes(IxDyn(&[0, 2, 1])).to_owned();
            b_t_shape = b_t.shape().to_vec();
        } else if b_shape.len() == 4 {
            todo!();
        } else {
            panic!("Matrix multiplication grad only supported for <=3D tensors");
        }
        return Tensor {
            id: make_node_uid(),
            data: b_t.iter().map(|x| *x).collect(),
            shape: b_t_shape,
            ..self.clone()
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        fn print_recursive<T: std::fmt::Debug>(f: &mut std::fmt::Formatter, shape: &[usize], array: &[T], idx: &mut usize, depth: usize) -> std::fmt::Result {
            if depth == shape.len() {
                write!(f, "{:?} ", array[*idx])?;
                *idx += 1;
            } else {
                let size = shape[depth];
                write!(f, "[")?;
                for i in 0..size {
                    print_recursive(f, shape, array, idx, depth + 1)?;
                    if i < size - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")?;
            }
            Ok(())
        }

        write!(f, "Tensor<{}>(", print_type_of(&self.data[0]))?;
        let mut idx = 0;
        print_recursive(f, &self.shape, &self.data, &mut idx, 0)?;
        write!(f, ", shape=")?;
        write!(f, "{:?})", self.shape)?;
        write!(f, "\n")?; 
        Ok(())
    }
}
