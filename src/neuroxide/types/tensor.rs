use std::fs::write;

use crate::types::device::Device;
use petgraph::{prelude::GraphMap, Directed};
use crate::utils::node_uid::makeNodeUID;

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    device: Device,
    requires_grad: bool,
    pub op_chain: GraphMap<i32, i32, Directed>,
    pub op_head: i32
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor<T> {
        let mut graph = GraphMap::new();
        let id = makeNodeUID();
        graph.add_node(id);
        Tensor {
            data,
            shape,
            device,
            requires_grad,
            op_chain: graph,
            op_head: id
        }
    }
}

fn print_type_of<T>(_: &T) -> &'static str {
    std::any::type_name::<T>()
}

impl<T: std::fmt::Debug> std::fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let mut count = 0;
        let mut shape_index = 0;
        let mut shape_reversed = self.shape.iter().rev().collect::<Vec<_>>();
        write!(f, "Tensor ({})(dtype::{}) \n[", self.device, print_type_of(&self.data[0]))?;
        for i in &self.data {
            if count <= 0 {
                write!(f, "[")?;
            }
            if count < shape_reversed[shape_index] - 1 {
                write!(f, "{:?} ", i)?;
            } else {
                write!(f, "{:?}", i)?;
                write!(f, "]")?;
                count = 0;
                shape_index += 1;
                
                // if shape_index < shape_reversed.len() {
                //     write!(f, "\n ")?;
                // }
            }
            count += 1;
        }
        write!(f, "]")?;
        return Ok(())
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

        let result_id = makeNodeUID();
        result_graph.add_node(result_id);
        result_graph.add_edge(result_id, self.op_head, 0);
        result_graph.add_edge(result_id, other.op_head, 0);

        Tensor {
            data: result,
            shape: self.shape.clone(),
            device: self.device,
            requires_grad: self.requires_grad || other.requires_grad,
            op_chain: result_graph,
            op_head: result_id
        }
    }
}
