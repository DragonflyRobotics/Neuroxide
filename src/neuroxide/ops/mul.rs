use num::NumCast;
use petgraph::prelude::GraphMap;
use crate::ops::op_generic::{Ops, Operation};
use crate::types::tensor::Tensor;
use crate::utils::node_uid::make_node_uid;
use std::ops::{Add, Mul};

#[derive(Debug, Clone)]
pub struct MulOp;

impl<T> Operation<T> for MulOp
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default + std::fmt::Debug + NumCast
{
    fn forward(inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);
        assert!(inputs[0].shape == inputs[1].shape);
        assert!(inputs[0].device == inputs[1].device);
        assert!(inputs[0].dtype.read().unwrap().get_dtype() == inputs[1].dtype.read().unwrap().get_dtype());
        let t = inputs[0].clone() * inputs[1].clone();
        let db = inputs[0].dtype.clone();
        db.write().unwrap().insert(t.clone());
        drop(db);
        t
    }

    fn backward(inputs: &Vec<&Tensor<T>>, grad: Option<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);
        // println!("Backward called on AddOp");
        // println!("Inputs: {:?}", inputs);
        // println!("Grad: {:?}", grad);

        //get index of grad in inputs without for loop
        let grad_index = inputs.iter().position(|&x| x.id == grad.unwrap().id).unwrap();

        
        return inputs[1 - grad_index].clone(); 
    }

    // fn clone_box(&self) -> Box<dyn Operation<T>> {
    //     Box::new(MulOp)
    // }
}

//implement add for tensor

impl<T> std::ops::Mul for Tensor<T>
where
    T: std::ops::Add<Output = T> + std::ops::Mul<Output = T> + Copy + Default,
{
    type Output = Tensor<T>;

    fn mul(self, other: Tensor<T>) -> Tensor<T> {
        assert!(self.shape == other.shape);
        assert!(self.device == other.device);
        let result = self.data.iter().zip(other.data.iter()).map(|(a, b)| *a * *b).collect();

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

        Tensor {
            id: result_id,
            data: result,
            shape: self.shape.clone(),
            device: self.device,
            op: Ops::MulEnum,
            requires_grad: self.requires_grad || other.requires_grad,
            op_chain: result_graph,
            op_head: result_id,
            dtype: self.dtype.clone()
        }
    }
}
