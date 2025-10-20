use neuroxide::{
    ops::{add::Add, mul::Mul, op::Operation},
    types::{tensor::Tensor, tensor_element::TensorHandleExt},
};

extern crate neuroxide;

// TODO: Fix benchmarks
// TODO: Replace NDArray (Maybe)
// TODO: Create Union Graph for operations on CUDA
// TODO: Enable caching/saving
// TODO: Add more operations
fn main() {
    let x = Tensor::new([1.0, 2.0, 3.0, 4.0], [2usize, 2usize]);

    // (x * x)
    let x2 = Mul::forward((x.clone(), x.clone()));
    // (x * (x + x))
    let x_plus_x = Add::forward((x.clone(), x.clone()));
    let x_xplusx = Mul::forward((x.clone(), x_plus_x.clone()));
    // (x*x + x)
    let x2_plus_x = Add::forward((x2.clone(), x.clone()));
    // y = (x*x + x) * (x * (x + x))
    let y = Mul::forward((x2_plus_x.clone(), x_xplusx.clone()));

    // Trigger backward pass
    y.backward();

    println!("Forward result y: {:?}", y);
    println!("Gradient dy/dx: {:?}", x.get_gradient());
}
