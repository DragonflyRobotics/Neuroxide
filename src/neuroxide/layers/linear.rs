use std::sync::{Arc, RwLock};

use crate::ops::op_generic::Operation;
use crate::ops::add::AddOp;
use crate::ops::matmul::MatMulOp;
use crate::types::T::TensorElement;
use crate::types::{device::Device, tensor::Tensor, tensordb::TensorDB};


pub struct Linear<T> {
    input_features: usize,
    output_features: usize,
    useBias: bool,
    weights: Tensor<T>,
    bias: Option<Tensor<T>>,
}

impl<T> Linear<T> 
where
    T: TensorElement
{
    pub fn new(db: Arc<RwLock<TensorDB<T>>>, input_features: usize, output_features: usize, use_bias: bool) -> Self {
        let mut weights = Tensor::<T>::new_uniform(&db, vec![input_features, output_features], Device::CPU, true);
        let mut bias = if use_bias {
            Some(Tensor::<T>::new_uniform(&db, vec![output_features], Device::CPU, true))
        } else {
            None
        };
        Linear {
            input_features,
            output_features,
            useBias: use_bias,
            weights,
            bias,
        }
    }

    pub fn forward(&self, input: Tensor<T>) -> Tensor<T> {
        match self.bias {
            Some(ref b) => {
                AddOp::forward(&vec![&MatMulOp::forward(&vec![&input, &self.weights]), b])
            }
            None => {
                MatMulOp::forward(&vec![&input, &self.weights])
            }
        }
    }
}
