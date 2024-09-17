use std::collections::HashMap;

use crate::{ops::{add::AddOp, mul::MulOp, op_generic::{Operation, Ops}}, types::device::Device};
use num::NumCast;
use petgraph::{algo, prelude::GraphMap, Directed, Direction::Outgoing};
use crate::utils::node_uid::make_node_uid;

use super::tensordb::TensorDB;

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub id: i32,
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub device: Device,
    pub op: Ops,
    pub requires_grad: bool,
    pub op_chain: GraphMap<i32, i32, Directed>,
    pub op_head: i32
}

impl<T> Tensor<T> 
where
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + Default + std::fmt::Debug + NumCast
{
    pub fn new(db: &mut TensorDB<T>, data: Vec<T>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor<T> {
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
            op_head: id
        };
        db.insert(t.clone());
        t
    }

    fn match_ops(&self, db: &mut TensorDB<T>, d: &Tensor<T>, dx: &Tensor<T>, inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        match d.op {
            Ops::AddEnum => {
                AddOp.backward(db, inputs, Some(dx))
            },
            Ops::MulEnum => {
                MulOp.backward(db, inputs, Some(dx))
            },
            _ => panic!("Operation not implemented")
        }
    }

    pub fn backward(&self, db: &mut TensorDB<T>, dx: Option<Vec<i32>>) -> HashMap<i32, Tensor<T>> {
        println!("Backward called on tensor: {:?}", self.id);
        let mut all_leaves = Vec::new();
        match dx {
            Some(x) => {
                for node in x {
                    all_leaves.push(node);
                }
            },
            None => {
                for node in self.op_chain.nodes() {
                    let outgoing_edges = self.op_chain.edges_directed(node, Outgoing);
                    if outgoing_edges.count() == 0 {
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
            let new_tensor = Tensor {
                id: self.id,
                data: data,
                shape: self.shape.clone(),
                device: self.device.clone(),
                op: self.op.clone(),
                requires_grad: self.requires_grad,
                op_chain: self.op_chain.clone(),
                op_head: self.id,
            };
            grad.insert(leaf, new_tensor);
        }

        for leaf in all_leaves.clone() {
            let path = &paths[&leaf];
            let mut arr: Vec<Tensor<T>> = Vec::new();
            for p in path {
                let mut temp = grad[&leaf].clone();
                for i in 0..p.len() - 1 {
                    // println!("d{:?}/d{:?}", p[i], p[i + 1]);
                    let neighbor = self.op_chain.neighbors_directed(p[i], Outgoing).collect::<Vec<_>>();
                    // println!("{:?} = {:?} + {:?}", p[i], neighbor[0], neighbor[1]);
                    let inputs = vec![db.get(neighbor[0]).unwrap(), db.get(neighbor[1]).unwrap()];
                    let output = self.match_ops(&mut db.clone(), db.get(p[i]).unwrap(), db.get(p[i+1]).unwrap(), &inputs);
                    temp = temp * output;
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

fn print_type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
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
