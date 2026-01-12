use std::fmt;
use std::sync::MutexGuard;
use std::{
    fmt::{Display, Formatter},
    sync::{Arc, Mutex},
};

use num::{Num, NumCast};
use trait_set::trait_set;

use crate::tensor::Tensor;

trait_set! {
    pub trait TensorElement = std::ops::Add<Output = Self> + Num + NumCast + Copy + Clone + std::fmt::Debug + std::fmt::Display + 'static + Send + Sync;
}

pub type SharedTensor<T> = Arc<Mutex<Tensor<T>>>;
pub trait TensorHandleExt<T> {
    fn lock_ref(&self) -> std::sync::MutexGuard<'_, Tensor<T>>;
    fn values_clone(&self) -> Vec<T>
    where
        T: Clone;
    fn backward(&self);
    fn get_gradient(&self) -> Option<SharedTensor<T>>;
    fn get_shape(&self) -> Box<[usize]>;
    fn get_stride(&self) -> Box<[usize]>;
}
impl<T: TensorElement> TensorHandleExt<T> for Arc<Mutex<Tensor<T>>> {
    fn lock_ref(&self) -> std::sync::MutexGuard<'_, Tensor<T>> {
        self.lock().unwrap()
    }

    fn values_clone(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.lock().unwrap().get_values().clone()
    }

    fn backward(&self) {
        self.lock().unwrap().backward();
    }
    fn get_gradient(&self) -> Option<SharedTensor<T>> {
        self.lock().unwrap().get_gradient()
    }
    fn get_shape(&self) -> Box<[usize]> {
        self.lock().unwrap().get_shape()
    }
    fn get_stride(&self) -> Box<[usize]> {
        self.lock().unwrap().get_stride()
    }
}
