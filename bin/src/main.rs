use neuroxide::{
    ops::{add::Add, op::Operation},
    types::tensor::Tensor,
};

extern crate neuroxide;

// TODO: Fix benchmarks
// TODO: Replace NDArray (Maybe)
// TODO: Create Union Graph for operations on CUDA
// TODO: Enable caching/saving
// TODO: Add more operations

fn main() {
    let a = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], Box::new([2, 2]));
    let b = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0], Box::new([2, 2]));
    println!("{:?}", a);
    println!("{:?}", b);
    let c = Add::forward(Box::new([a.clone(), a.clone()]));
    let d = Add::forward(Box::new([c.clone(), a.clone()]));
    println!("{:?}", d);
    d.lock().unwrap().backward();
    println!("{:?}", a.lock().unwrap().get_gradient());
}
