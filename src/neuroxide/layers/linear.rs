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
    pub weights: Tensor<T>,
    pub bias: Option<Tensor<T>>,
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
            weights,
            bias,
        }
    }

    pub fn forward(&mut self, input: &Tensor<T>) -> Tensor<T> {
        println!("{:?}", self.weights.data);
        self.weights = self.db.read().unwrap().get(self.weights.id).unwrap().clone();
        if let Some(ref b) = self.bias {
            self.bias = Some(self.db.read().unwrap().get(b.id).unwrap().clone());
        }
        match self.bias {
            Some(ref b) => {
                AddOp::forward(&vec![&MatMulOp::forward(&vec![input, &self.weights]), b])
            }
            None => {
                MatMulOp::forward(&vec![input, &self.weights])
            }
        }
    }

    pub fn parameters(&self) -> HashMap<String, i32> {
        let mut params = HashMap::new();
        params.insert(format!("linear_{}_weights", self.id), self.weights.id);
        if let Some(ref b) = self.bias {
            params.insert(format!("linear_{}_bias", self.id), b.id);
        }
        params
    }
}
