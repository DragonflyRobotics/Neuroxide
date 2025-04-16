use num::{Num, NumCast};
use trait_set::trait_set;

use crate::ops::f_to_i_ops::{CosOpTrait, LnOpTrait, PowOpTrait, SinOpTrait};

trait_set! {
    pub trait TensorElement = std::ops::Add<Output = Self> + std::ops::Mul<Output = Self> + Copy + Default + std::fmt::Debug + NumCast + SinOpTrait + CosOpTrait + PowOpTrait + LnOpTrait + Num + ndarray::ScalarOperand + ndarray::LinalgScalar
}
