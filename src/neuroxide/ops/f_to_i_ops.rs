pub trait SinOpTrait {
    fn sin(self) -> Self;
}

pub trait CosOpTrait {
    fn cos(self) -> Self;
}

//--------------------

impl SinOpTrait for i8 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).sin() as i8
    }
}

impl SinOpTrait for i16 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).sin() as i16
    }
}

impl SinOpTrait for i32 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).sin() as i32
    }
}

impl SinOpTrait for i64 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).sin() as i64
    }
}

impl SinOpTrait for i128 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).sin() as i128
    }
}

impl SinOpTrait for u8 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).sin() as u8
    }
}

impl SinOpTrait for u16 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).sin() as u16
    }
}

impl SinOpTrait for u32 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).sin() as u32
    }
}

impl SinOpTrait for u64 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).sin() as u64
    }
}

impl SinOpTrait for u128 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).sin() as u128
    }
}

impl SinOpTrait for f32 {
    fn sin(self) -> Self {
        self.sin()
    }
}

impl SinOpTrait for f64 {
    fn sin(self) -> Self {
        self.sin()
    }
}

impl CosOpTrait for i8 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).cos() as i8
    }
}

impl CosOpTrait for i16 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).cos() as i16
    }
}

impl CosOpTrait for i32 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).cos() as i32
    }
}

impl CosOpTrait for i64 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).cos() as i64
    }
}

impl CosOpTrait for i128 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).cos() as i128
    }
}

impl CosOpTrait for u8 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).cos() as u8
    }
}

impl CosOpTrait for u16 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).cos() as u16
    }
}

impl CosOpTrait for u32 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).cos() as u32
    }
}

impl CosOpTrait for u64 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).cos() as u64
    }
}

impl CosOpTrait for u128 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
        (self as f32).cos() as u128
    }
}

impl CosOpTrait for f32 {
    fn cos(self) -> Self {
        self.cos()
    }
}

impl CosOpTrait for f64 {
    fn cos(self) -> Self {
        self.cos()
    }
}   

