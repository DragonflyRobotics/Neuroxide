# Neuroxide
[![Rust](https://github.com/DragonflyRobotics/Neuroxide/actions/workflows/rust.yml/badge.svg)](https://github.com/DragonflyRobotics/Neuroxide/actions/workflows/rust.yml)

Welcome! This project attempts to rewrite the PyTorch framework (maintaining a consistent API call) in Rust in hopes of a faster, hard-typed AI framework. This project is currently in its Alpha phase, so feel free to contribute or contact me at my [email](kshahusa@gmail.com)! As this project is in its early phases, documentation will be sparse, but a quick overview of the development scope will be provided below.

## News
A major part of the timeline (MatMul) is completed, tested, and benchmarked. With this, some primitive broadcasting for multiplication is here but is not recommended. Users are recommended to still manually broadcast any constant terms to the size of argument size. Here is a sample of matmul and gradient in action!

```rust
use std::sync::{Arc, RwLock};

use neuroxide::{ops::{add::AddOp, matmul::MatMulOp, mul::MulOp, pow::PowOp}, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use neuroxide::ops::op_generic::Operation;

fn main() {
    let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0];
    let b: Vec<f32> = vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let a_shape = vec![2, 2, 2];
    let b_shape = vec![2, 2, 3];

    let tensor_db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let c1 = Tensor::new(&tensor_db, vec![6.0; 12], vec![2, 2, 3], Device::CUDA, true);
    let c2 = Tensor::new(&tensor_db, vec![5.0; 12], vec![2, 2, 3], Device::CUDA, true);
    let c3 = Tensor::new(&tensor_db, vec![2.0; 12], vec![2, 2, 3], Device::CUDA, true);


    let a = Tensor::new(&tensor_db, a, a_shape, Device::CUDA, true);
    let b = Tensor::new(&tensor_db, b, b_shape, Device::CUDA, false);
    let c = MatMulOp::forward(&vec![&a, &b]);
    let d = MulOp::forward(&vec![&c, &c1]);
    let e = AddOp::forward(&vec![&d, &c2]);
    let f = PowOp::forward(&vec![&e, &c3]);
    println!("{}", f);
    let grad = f.backward(None);
    println!("{}", grad.get(&a.id).unwrap());
}
```

## Table of Contents
- [Usage](#usage)
- [Sample Code](#sample-code)
- [Goal](#goal)
- [Contributing](#contributing)

## Usage
Here is how a contributor/developer might use the project.
1. `git clone git@github.com:DragonflyRobotics/Neuroxide.git`
2. Modify the `src/bin.rs` to contain your personal programs
3. `cargo run` 

### External Use (expert)
You must compile the library via `cargo build` and copy the file from the `target` folder. You can then link this to your Rust projects to use. You can also try installing like this:
`cargo install --git git@github.com:DragonflyRobotics/Neuroxide.git`

## Sample Code
Here are some basic operations (we hope you see the similarity to PyTorch):

_Forward Pass_
```rust
let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
let mut c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
let mut c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
let mut result = AddOp::forward(&vec![&c1c, &c2c]);
```

_Backward Pass_
```rust
let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
let x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
let c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CPU, false);
let c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, false);
let r1 = MulOp::forward(&vec![&x, &c1c]);
let r2 = MulOp::forward(&vec![&x, &c2c]);
let mut result = AddOp::forward(&vec![&r1, &r2]);
result = MulOp::forward(&vec![&result, &x]);
println!(result.data[0], 525.0));

let grad = result.backward(None);
println!(grad.get(&x.id).unwrap().data[0])
```

_Forward Pass CUDA_
```rust
let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
let mut c1c = Tensor::new(&db, vec![15.0], vec![1], Device::CUDA, false);
let mut c2c = Tensor::new(&db, vec![6.0], vec![1], Device::CUDA, false);
let mut result = AddOp::forward(&vec![&c1c, &c2c]);
```

_Partial Backward to Selective Leaves_
```rust
let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
let x1 = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
let x2 = Tensor::new(&db, vec![6.0], vec![1], Device::CPU, true);
let x3 = Tensor::new(&db, vec![7.0], vec![1], Device::CPU, true);
let x4 = Tensor::new(&db, vec![8.0], vec![1], Device::CPU, true);


let result = x1.clone() * (x2.clone() + x3) + x4;
println!(result.data[0]);
let grad = result.backward(Some(vec![x2.id.clone()]));
println!(grad.get(&x2.id).unwrap().data[0]);
```
**Note:** You can avoid the clunky notation and simply operate on tensors using `+`, `-`, `*`, and `/`! 

_Simple Neural Network_
```rust
extern crate blas_src;
use std::{sync::{Arc, RwLock}, time::{SystemTime, UNIX_EPOCH}};

use neuroxide::{layers::linear::Linear, types::{device::Device, tensor::Tensor, tensordb::{DTypes, TensorDB}}};
use neuroxide::ops::op_generic::Operation;
use neuroxide::optimizers::simple_descent::SimpleDescent;
use rand::Rng;

#[macro_use]
extern crate neuroxide;

fn main() {
    let start = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let pow_const = Tensor::<f32>::new(&db, vec![2.0; 1], vec![1], Device::CPU, false);
    let mut linear1 = Linear::new(&db, 16, 16, true);
    let mut linear2 = Linear::new(&db, 16, 16, true);
    let mut optim = SimpleDescent::new(&db, 0.0000001);
    optim.add_parameters(&linear1.parameters());
    optim.add_parameters(&linear2.parameters());
    for iteration in 0..1500 {
        let num: f32 = rand::thread_rng().gen_range(0..100) as f32; 
        let input = Tensor::<f32>::new(&db, vec![num; 16], vec![1, 16], Device::CPU, false);
        let output = Tensor::<f32>::new(&db, vec![num * 2.0; 16], vec![1, 16], Device::CPU, false);


        let mut c = linear1.forward(&input);
        c = linear2.forward(&c);

        let loss = pow!(c - output, pow_const);
        let grad = loss.backward(None);

        optim.step(&grad);

        println!("Epoch: {} Loss: {}", iteration, loss);
    } 

    let input = Tensor::<f32>::new(&db, vec![4.0; 16], vec![16], Device::CPU, false);

    let c = linear1.forward(&input);
    let c = linear2.forward(&c);
    println!("{}", c);


    let end = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    println!("Time taken: {:?} seconds", end-start);
}
```
**Note:** The layers are manually implemented here but built-in `neuroxide.nn.Linear` functionality is coming soon!

## Goal
Python has many benefits, mainly its flexibility, which makes it an avid language for AI/ML. The tradeoff is the clunky interpreter, alternation between Python and C++ bindings, and lack of multiprocessing, which make it inefficient and slow for many high-performance applications. This project attempts to maintain the comforts of the PyTorch syntax while leveraging a hard-typed, efficient language to create a powerful AI engine for cutting-edge projects. 

## Contributing
We appreciate any contributions to this project to help it grow and encompass the full functionality of an AI engine. Please refer to our [contributing guidelines](https://github.com/DragonflyRobotics/Neuroxide/blob/dev/CONTRIBUTING.md) for details. 

## License
This project has a GNU License, which can be found [here](https://github.com/DragonflyRobotics/Neuroxide/blob/dev/LICENSE).
