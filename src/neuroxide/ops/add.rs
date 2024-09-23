use num::NumCast;
use petgraph::prelude::GraphMap;
use crate::ops::op_generic::{Ops, Operation};
use crate::types::tensor::Tensor;
use crate::types::tensordb::TensorDB;
use crate::utils::node_uid::make_node_uid;
use std::ops::{Add, Mul};

#[derive(Debug, Clone)]
pub struct AddOp;

impl<T> Operation<T> for AddOp
where
    T: Add<Output = T> + Mul<Output = T> + Copy + Default + std::fmt::Debug + NumCast
{
    fn forward(&self, inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);
        let t = inputs[0].clone() + inputs[1].clone();
        let db = inputs[0].dtype.clone();
        db.write().unwrap().insert(t.clone());
        drop(db);
        t
    }

    fn backward(&self, inputs: &Vec<&Tensor<T>>, grad: Option<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 2);
        // println!("Backward called on AddOp");
        // println!("Inputs: {:?}", inputs);
        // println!("Grad: {:?}", grad);
        let mut grad_data = vec![T::default(); inputs[0].data.len()];
        if inputs[0].id == inputs[1].id { //c = a + a => dc/da = 2
            for i in 0..inputs[0].data.len() {
                grad_data[i] = T::from(2).unwrap(); 
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

    fn clone_box(&self) -> Box<dyn Operation<T>> {
        Box::new(AddOp)
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
            op: Ops::AddEnum,
            requires_grad: self.requires_grad || other.requires_grad,
            op_chain: result_graph,
            op_head: result_id,
            dtype: self.dtype.clone()
        }
    }
}
