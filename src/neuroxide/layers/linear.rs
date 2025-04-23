use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::ops::op_generic::Operation;
use crate::ops::add::AddOp;
use crate::ops::matmul::MatMulOp;
use crate::types::T::TensorElement;
use crate::types::{device::Device, tensor::Tensor, tensordb::TensorDB};
use crate::utils::node_uid::make_node_uid;


pub struct Linear<T> {
    id: i32,
    db: Arc<RwLock<TensorDB<T>>>,
    input_features: usize,
    output_features: usize,
    useBias: bool,
    pub weights: Arc<RwLock<Tensor<T>>>,
    pub bias: Option<Arc<RwLock<Tensor<T>>>>,
}

impl<T> Linear<T> 
where
    T: TensorElement
{
    pub fn new(db: &Arc<RwLock<TensorDB<T>>>, input_features: usize, output_features: usize, use_bias: bool) -> Self {
        let weights = Tensor::<T>::new_uniform(db, vec![input_features, output_features], Device::CPU, true);
        let bias = if use_bias {
            Some(Tensor::<T>::new_uniform(db, vec![output_features], Device::CPU, true))
        } else {
            None
        };
        Linear {
            id: make_node_uid(),
            db: db.clone(),
            input_features,
            output_features,
            useBias: use_bias,
            weights: Arc::new(RwLock::new(weights)),
            bias: bias.map(|b| Arc::new(RwLock::new(b))),
        }
    }

    pub fn forward(&mut self, input: &Tensor<T>) -> Tensor<T> {
        let weights = self.weights.read().unwrap();
        match self.bias.clone() {
            Some(b) => {
                AddOp::forward(&vec![&MatMulOp::forward(&vec![input, &weights]), &b.read().unwrap()])
            }
            None => {
                MatMulOp::forward(&vec![input, &weights])
            }
        }
    }
    
    pub fn parameters(&self) -> HashMap<String, Arc<RwLock<Tensor<T>>>> {
        let mut params = HashMap::new();
        params.insert(format!("linear_{}_weights", self.id), self.weights.clone());
        if let Some(b) = self.bias.clone() {
            params.insert(format!("linear_{}_bias", self.id), b);
        }
        params
    }
}
