use std::{collections::HashMap, sync::{Arc, RwLock}};

use crate::types::{tensor::Tensor, tensordb::TensorDB, T::TensorElement};

pub struct SimpleDescent<T> {
    db: Arc<RwLock<TensorDB<T>>>,
    lr: Tensor<T>,
    parameters: Vec<i32>,
}

impl <T> SimpleDescent<T> 
where
    T: TensorElement
{
    pub fn new(db: &Arc<RwLock<TensorDB<T>>>, lr: Tensor<T>) -> Self {
        assert!(lr.shape == vec![1], "Learning rate must be a scalar");
        SimpleDescent {
            db: db.clone(),
            lr,
            parameters: Vec::new(),
        }
    }

    pub fn add_parameters(&mut self, params: &HashMap<String, i32>) {
        for param in params.values() {
            self.parameters.push(*param);
        }
    }

    pub fn step(&mut self, grad: &HashMap<i32, Tensor<T>>) {
        for param_id in &mut self.parameters {
            let p = self.db.read().unwrap().get(*param_id).unwrap().clone();
            let grad_param = grad.get(&p.id).unwrap().clone();
            let new_param = p.clone() - (grad_param * self.lr.clone());
            let mut db = self.db.try_write().unwrap();
            let param: &mut Tensor<T> = db.get_mut(*param_id).unwrap();
            param.data = new_param.data;
            param.shape = new_param.shape;
            param.clear_graph();
        }
    }
}
