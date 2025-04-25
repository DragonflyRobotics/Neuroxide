use std::{collections::HashMap, sync::{Arc, RwLock}};

use crate::types::{tensor::Tensor, tensordb::TensorDB, T::TensorElement};

pub struct SimpleDescent<T> {
    db: Arc<RwLock<TensorDB<T>>>,
    lr: Tensor<T>,
    parameters: Vec<Arc<RwLock<Tensor<T>>>>,
}

impl <T> SimpleDescent<T> 
where
    T: TensorElement
{
    pub fn new(db: &Arc<RwLock<TensorDB<T>>>, lr: f32) -> Self {
        SimpleDescent {
            db: db.clone(),
            lr: Tensor::<T>::new(db, vec![T::from(lr).unwrap()], vec![1], crate::types::device::Device::CPU, false),
            parameters: Vec::new(),
        }
    }

    pub fn add_parameters(&mut self, params: &HashMap<String, Arc<RwLock<Tensor<T>>>>) {
        for param in params.values() {
            self.parameters.push(param.clone());
        }
    }

    pub fn step(&mut self, grad: &HashMap<i32, Tensor<T>>) {
        for param in &mut self.parameters {
            let param_id = param.read().unwrap().id;
            let mut p = param.write().unwrap();
            p.cpu();

            let grad_param = grad.get(&param_id).unwrap().clone();
            let new_param = p.clone() - (grad_param * self.lr.clone());
            p.data = new_param.data;
            p.shape = new_param.shape;
            p.clear_graph();
            p.cuda();
        }
    }
}
