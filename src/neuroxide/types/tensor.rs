use std::{collections::HashMap, sync::{Arc, RwLock}};

use crate::{ops::{add::AddOp, f_to_i_ops::{CosOpTrait, SinOpTrait}, mul::MulOp, op_generic::{Operation, Ops}, sin::SinOp}, types::device::Device, utils::types::print_type_of};
use num::{Num, NumCast};
use petgraph::{algo, prelude::GraphMap, Directed, Direction::Outgoing};
use crate::utils::node_uid::make_node_uid;

use super::tensordb::{assert_types, TensorDB};

#[derive(Clone)]
pub struct Tensor<T> {
    pub id: i32,
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub device: Device,
    pub op: Ops,
    pub requires_grad: bool,
    pub op_chain: GraphMap<i32, i32, Directed>,
    pub op_head: i32,
    pub dtype: Arc<RwLock<TensorDB<T>>>
}

impl<T> Tensor<T> 
where
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + Default + std::fmt::Debug + NumCast + SinOpTrait + CosOpTrait + Num
{
    pub fn new(db: &Arc<RwLock<TensorDB<T>>>, data: Vec<T>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor<T> {
        assert_types(db.read().unwrap().get_dtype(), data[0]);
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

    fn match_ops(&self, d: &Tensor<T>, dx: &Tensor<T>, inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        match d.op {
            Ops::AddEnum => {
                AddOp::backward(inputs, Some(dx))
            },
            Ops::MulEnum => {
                MulOp::backward(inputs, Some(dx))
            },
            Ops::SinEnum => {
                SinOp::backward(inputs, Some(dx))
            },
            _ => panic!("Operation not implemented")
        }
    }

    pub fn backward(&self, dx: Option<Vec<i32>>) -> HashMap<i32, Tensor<T>> {
        let mut all_leaves = Vec::new();
        let db = self.dtype.read().unwrap();
        match dx {
            Some(x) => {
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
                let mut temp = grad[&leaf].clone();
                grad.get_mut(&leaf).unwrap().op_head = grad.get(&leaf).unwrap().id;
                for i in 0..p.len() - 1 {
                    // println!("d{:?}/d{:?}", p[i], p[i + 1]);
                    let neighbor = self.op_chain.neighbors_directed(p[i], Outgoing).collect::<Vec<_>>();
                    // println!("{:?} = {:?} + {:?}", p[i], neighbor[0], neighbor[1]);
                    let mut inputs = vec![];
                    for n in neighbor {
                        inputs.push(db.get(n).unwrap());
                    }
                    // let inputs = vec![db.get(neighbor[0]).unwrap(), db.get(neighbor[1]).unwrap()];
                    let output = self.match_ops(db.get(p[i]).unwrap(), db.get(p[i+1]).unwrap(), &inputs);
                    temp = temp * output;
                    grad.get_mut(&leaf).unwrap().op_chain.add_edge(p[i], p[i + 1], 0);
                }

                arr.push(temp);
            }
            let mut sum = arr[0].clone();
            for i in 1..arr.len() {
                sum = sum + arr[i].clone();
            }
            grad.get_mut(&leaf).unwrap().data = sum.data;
        }
        grad
    }

    pub fn clear_graph(&mut self) {
        self.op_chain = GraphMap::new();
        self.op_chain.add_node(self.id);
        self.op_head = self.id;
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
        write!(f, "\n")?; 
        Ok(())
    }
}
