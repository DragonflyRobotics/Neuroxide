trait SinOpTrait {
    fn sin_op(self) -> Self;
}

impl SinOpTrait for f64 {
    fn sin_op(self) -> f64 {
        self.sin()
    }
}

fn main() {
    let x = 3.14/4.0;
    let y = x.sin_op();
    println!("sin({}) = {}", x, y);
}

