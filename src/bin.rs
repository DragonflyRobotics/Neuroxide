use neuroxide::ops::add::AddOp;
use neuroxide::ops::op_generic::Operation;
use neuroxide::types::tensor::Tensor;
use neuroxide::types::device::Device;
use petgraph::dot::Dot;


fn main() {
    let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![4, 1], Device::CPU, true);
    let b = Tensor::new(vec![4.0, 5.0, 6.0, 7.0], vec![2, 2], Device::CPU, true);
    let c = Tensor::new(vec![4.0, 5.0, 6.0, 7.0], vec![2, 2], Device::CPU, true);
    // let mut result = AddOp.forward(&vec![&a, &b]);
    // result = AddOp.forward(&vec![&result, &b]);
    // println!("{:?}", result);
    // println!("{:?}", Dot::new(&result.op_chain))
    println!("{}", a);
}
