# Neuroxide
[![Rust](https://github.com/DragonflyRobotics/Neuroxide/actions/workflows/rust.yml/badge.svg)](https://github.com/DragonflyRobotics/Neuroxide/actions/workflows/rust.yml)

Welcome! This project attempts to rewrite the PyTorch framework (maintaining a consistent API call) in Rust in hopes of a faster, hard-typed AI framework. This project is currently in its Alpha phase, so feel free to contribute or contact me at my [email](kshahusa@gmail.com)! As this project is in its early phases, documentation will be sparse, but a quick overview of the development scope will be provided below.

## Table of Contents
- [About](#about)
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


## Goal
Python has many benefits, mainly its flexibility, which makes it an avid language for AI/ML. The tradeoff is the clunky interpreter, alternation between Python and C++ bindings, and lack of multiprocessing, which make it inefficient and slow for many high-performance applications. This project attempts to maintain the comforts of the PyTorch syntax while leveraging a hard-typed, efficient language to create a powerful AI engine for cutting-edge projects. 

## Contributing
We appreciate any contributions to this project to help it grow and encompass the full functionality of an AI engine. Please refer to our [contributing guidelines](https://github.com/DragonflyRobotics/Neuroxide/blob/dev/CONTRIBUTING.md) for details. 

## License
This project has a GNU License, which can be found [here](https://github.com/DragonflyRobotics/Neuroxide/blob/dev/LICENSE).
