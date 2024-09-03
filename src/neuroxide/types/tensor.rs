use crate::{ops::op_generic::Operation, types::device::Device};
use petgraph::{data::Build, graph::{DiGraph, NodeIndex}, prelude::GraphMap};
use petgraph::dot::Dot;

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    data: Vec<T>,
    shape: Vec<usize>,
    device: Device,
    requires_grad: bool,
    pub op_chain: GraphMap<NodeIndex, (), petgraph::Directed>,
}

impl<T> Tensor<T> {
    pub fn new(data: Vec<T>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor<T> {
        Tensor {
            data,
            shape,
            device,
            requires_grad,
            op_chain: GraphMap::new(), 
        }
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
            result_graph.add_edge(edge.0, edge.1, ());
        }
        for edge in other_edges {
            result_graph.add_edge(edge.0, edge.1, ());
        }
        
        result_graph.add_node(NodeIndex::new(0));
        result_graph.add_edge(, b, weight)

        println!("{:?}", Dot::new(&result_graph));
        Tensor {
            data: result,
            shape: self.shape.clone(),
            device: self.device,
            requires_grad: self.requires_grad || other.requires_grad,
            op_chain: result_graph,
        }
    }
}
