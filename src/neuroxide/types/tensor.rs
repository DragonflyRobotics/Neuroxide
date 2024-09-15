use std::collections::{HashMap, HashSet};

use crate::types::device::Device;
use petgraph::{algo, prelude::GraphMap, Directed, Direction::{Incoming, Outgoing}};
use crate::utils::node_uid::make_node_uid;

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    pub id: i32,
    data: Vec<T>,
    shape: Vec<usize>,
    device: Device,
    grad: Option<GraphMap<i32, i32, Directed>>,
    requires_grad: bool,
    pub op_chain: GraphMap<i32, i32, Directed>,
    pub op_head: i32
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor<T> {
        let mut graph = GraphMap::new();
        let id = make_node_uid();
        graph.add_node(id);
        Tensor {
            id,
            data,
            shape,
            device,
            grad: None,
            requires_grad,
            op_chain: graph,
            op_head: id
        }
    }

    pub fn backward(&self, dx: Option<Vec<i32>>) {
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
        for leaf in all_leaves {
            let path = algo::all_simple_paths::<Vec<_>, _>(&self.op_chain, self.id, leaf, 0, None).collect::<Vec<_>>();
            paths.insert(leaf, path);
        }
        println!("All paths: {:?}", paths);

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



//implement add for tensor

impl<T> std::ops::Add for Tensor<T>
where
    T: std::ops::Add<Output = T> + Copy + Default,
{
    type Output = Tensor<T>;

    fn add(self, other: Tensor<T>) -> Tensor<T> {
        assert!(self.shape == other.shape);
        assert!(self.device == other.device);
        let result = self.data.iter().zip(other.data.iter()).map(|(a, b)| *a + *b).collect();

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
            result_graph.add_edge(edge.0, edge.1, 0);
        }
        for edge in other_edges {
            result_graph.add_edge(edge.0, edge.1, 0);
        }

        let result_id = make_node_uid();
        result_graph.add_node(result_id);
        result_graph.add_edge(result_id, self.op_head, 0);
        result_graph.add_edge(result_id, other.op_head, 0);

        Tensor {
            id: result_id,
            data: result,
            shape: self.shape.clone(),
            device: self.device,
            grad: None,
            requires_grad: self.requires_grad || other.requires_grad,
            op_chain: result_graph,
            op_head: result_id
        }
    }
}
