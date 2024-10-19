pub trait SinOpTrait {
    fn sin(self) -> Self;
}

pub trait CosOpTrait {
    fn cos(self) -> Self;
}

pub trait LnOpTrait {
    fn ln(self) -> Self;
}

pub trait PowOpTrait {
    fn pow(self, other: Self) -> Self;
}

//--------------------

impl SinOpTrait for i8 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl SinOpTrait for i16 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl SinOpTrait for i32 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl SinOpTrait for i64 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl SinOpTrait for i128 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl SinOpTrait for u8 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl SinOpTrait for u16 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl SinOpTrait for u32 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl SinOpTrait for u64 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl SinOpTrait for u128 {
    fn sin(self) -> Self {
        panic!("Not implemented for i8");
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
    }
}

impl CosOpTrait for i16 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl CosOpTrait for i32 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl CosOpTrait for i64 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl CosOpTrait for i128 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl CosOpTrait for u8 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl CosOpTrait for u16 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl CosOpTrait for u32 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl CosOpTrait for u64 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl CosOpTrait for u128 {
    fn cos(self) -> Self {
        panic!("Not implemented for i8");
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

impl PowOpTrait for i8 {
    fn pow(self, other: Self) -> Self {
        self.pow(other.try_into().unwrap())
    }
}

impl PowOpTrait for i16 {
    fn pow(self, other: Self) -> Self {
        self.pow(other.try_into().unwrap())
    }
}

impl PowOpTrait for i32 {
    fn pow(self, other: Self) -> Self {
        self.pow(other.try_into().unwrap())
    }
}

impl PowOpTrait for i64 {
    fn pow(self, other: Self) -> Self {
        self.pow(other.try_into().unwrap())
    }
}

impl PowOpTrait for i128 {
    fn pow(self, other: Self) -> Self {
        self.pow(other.try_into().unwrap())
    }
}

impl PowOpTrait for u8 {
    fn pow(self, other: Self) -> Self {
        self.pow(other.into())
    }
}

impl PowOpTrait for u16 {
    fn pow(self, other: Self) -> Self {
        self.pow(other.into())
    }
}

impl PowOpTrait for u32 {
    fn pow(self, other: Self) -> Self {
        self.pow(other)
    }
}

impl PowOpTrait for u64 {
    fn pow(self, other: Self) -> Self {
        panic!("Not implemented for u64");
    }
}

impl PowOpTrait for u128 {
    fn pow(self, other: Self) -> Self {
        panic!("Not implemented for u128");
    }
}

impl PowOpTrait for f32 {
    fn pow(self, other: Self) -> Self {
        self.powf(other)
    }
}

impl PowOpTrait for f64 {
    fn pow(self, other: Self) -> Self {
        self.powf(other)
    }
}

impl LnOpTrait for i8 {
    fn ln(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl LnOpTrait for i16 {
    fn ln(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl LnOpTrait for i32 {
    fn ln(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl LnOpTrait for i64 {
    fn ln(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl LnOpTrait for i128 {
    fn ln(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl LnOpTrait for u8 {
    fn ln(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl LnOpTrait for u16 {
    fn ln(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl LnOpTrait for u32 {
    fn ln(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl LnOpTrait for u64 {
    fn ln(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl LnOpTrait for u128 {
    fn ln(self) -> Self {
        panic!("Not implemented for i8");
    }
}

impl LnOpTrait for f32 {
    fn ln(self) -> Self {
        self.ln()
    }
}

impl LnOpTrait for f64 {
    fn ln(self) -> Self {
        self.ln()
    }
}

