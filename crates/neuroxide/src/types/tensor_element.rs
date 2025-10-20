use num::{Num, NumCast};
use trait_set::trait_set;

trait_set! {
    pub trait TensorElement = std::ops::Add<Output = Self> + Num + NumCast + Copy + Clone + std::fmt::Debug;
}
