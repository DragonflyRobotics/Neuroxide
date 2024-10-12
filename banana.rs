trait SinOpTrait {
    fn sin_op(self) -> Self;
}

impl SinOpTrait for f64 {
    fn sin_op(self) -> f64 {
        self.sin()
    }
}
impl SinOpTrait for i32{
    fn sin_op(self) -> i64 {
        panic!("i64 does not have sin method");
    }
}


fn main() {
    let x = 3;
    let y = x.sin_op();
    println!("sin({}) = {}", x, y);
}

