# Welcome to Neuroxide's Dev Team
Hello, we are pleased to see contributing interest in this project. Below, you will find all the necessary information for a newcomer or a seasoned professional to find the standards and contributing methods. 

As you contribute, please adhere to our [Code of Conduct](https://github.com/DragonflyRobotics/Neuroxide/blob/dev/CODE_OF_CONDUCT.md) and positively collaborate. 

## Newcomers
Firstly, you will need the basic build tools and SCM (source-control management). Please ensure you have the following installed:
* [Git-SCM](https://git-scm.com/)
* [Rust](https://www.rust-lang.org/) with the stable **AND** nightly toolchains (nightly is used only for benchmarking)
* CUDA and CuDNN (if you wish to use the NVidia features)

**Please Note:** Most features should work across platforms, but this project is primarily based and developed on Linux. As such, we highly recommend development on a Linux device or virtual machine.

## Project Organization 
```
├── src
│   ├── neuroxide
│   │   ├── Source files for the Neuroxide lib
│   ├── views
│   ├── model
│   ├── bin.rs // An executable file to test the Neuroxide lib
├── test
│   ├── Location of all the test cases for this project
├── bench
│   ├── Location of all the benchmarks for this project
├── Cargo.toml // Configuration file for compilation and dependencies
├── build.rs // Additional build script to bind CUDA libs and compile extra CUDA kernels with NVCC
```

## Ways to Contribute
There are many individual ways to contribute to this project that vary in difficulty and contribution level:

### Issues
The _Issues_ tab on GitHub is a way to notify developers of any bugs in the software or find issues that you can solve and contribute code to. 

**Submitting an Issue**
When submitting a new issue, please adhere to the template and provide as much detail about the error as possible. Here are potential things to include:
* The error logs and commands executed.
* Any software changes made
* Operating system/versions/hardware/drivers

**Finding/Resolving Issues**
This can be a great way to find smaller tasks to complete that fit individual skill sets. Here are the suggested steps:
1. Find a relevant issue you can solve. You can use the various tags and filters to find one you can best contribute to.
2. Resolve the issue on your personal **Fork** of this repository.
3. Push to your personal **Fork**.
4. Ensure your code passes all the tests, benchmarks, and coverage requirements. Additionally, ensure you adhere to the conventions, structure, and formatting of this project.
5. Request a PR (Pull Request) to merge your changes with the source code.
6. Wait for contributor approval and make changes as suggested by contributors.
7. Celebrate as you contribute to this project!

### Large Feature Implementations
Through the **Github Projects** tabs, you will see the many timelines, plans, and upcoming features planned for this project. You may choose to implement specific upcoming tasks and contribute to the project.
1. Use your personal **Fork** to implement the feature.
2. Develop test cases and benchmarks with at least 85% code coverage (if applicable).
3. Push to your **Fork** followed by a request to PR.
4. Wait for contributor approval as we audit the code quality, style, stability, and integration.

## Testing and Benchmarking Code
This is a critical component of the project to ensure accuracy and reliability for end users. Any contribution must pass all tests **if no new features were added** or implement new tests **if features were added**. 

### Testing
All tests are to be made in the `test` directory, and filenames are **always** prefixed with `test_`. Here is what one such test looks like:
```rust
#[test]
fn print_device() {
    assert!(Device::CPU.to_string() == "Device::CPU");
    assert!(Device::CUDA.to_string() == "Device::CUDA");
}
```

For numerical tests, please use the appropriate macros to adjust for floating-point inaccuracies:
```rust
assert!(relative_eq!(result.data[i], c1c.data[i] + c2c.data[i], epsilon = f64::EPSILON));
```

To run all the tests for CPU:
```
cargo test
```
and append the feature for CUDA:
```
cargo test --features cuda
```

### Benchmarks
All tests are to be made in the `bench` directory, and filenames are **always** prefixed with `bench_`. Here is what one such test looks like:
```rust
#[bench]
fn clear_graph(b: &mut test::Bencher) {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F64)));
    let mut x = Tensor::new(&db, vec![5.0], vec![1], Device::CPU, true);
    x.op_chain.add_node(1);
    x.op_chain.add_node(2);
    x.op_chain.add_edge(1, 2, 1);
    x.op_chain.add_edge(2, 1, 1);
    b.iter(|| {
        x.clear_graph();
    });
}
```
The function being benchmarked must be placed inside the closure inside `b.iter`, and `b` must be an argument to every test function. Please move all code **NOT** being tested outside the closure to ensure accurate measurements.

**Note:** Not all functions/features/fixes need benchmarking. Add benchmarks only if necessary or if requested by a contributor. 

Benchmarks must be run through the nightly toolchain as such for CPU:
```
cargo +nightly bench
```
and for CUDA:
```
cargo +nightly bench --features cuda
```

### Coverage Analysis
Coverage analysis helps ensure that all aspects of the software are adequately tested. Please refer to the amazing tool [Tarpaulin](https://github.com/xd009642/tarpaulin) for more instructions as that is our primary coverage analysis tool.

**Please Note:** Currently, coverage analysis only works on CUDA-enabled devices.

You must first install Tarpaulin:
```
cargo install tarpaulin
```

Then, run it on the project:
```
cargo tarpaulin --features cuda
```

# Thank You
Contributors make the Open Source Community thrive, and we sincerely appreciate any time, energy, effort, and dedication you have placed into this project. If you have any further questions, please do not hesitate to reach out to us!
