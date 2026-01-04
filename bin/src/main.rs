use core::time;

// use cuda::memory::cudadevice::CudaDevice;
use mempool::device_allocator::DeviceAllocator;
use mempool::pool::Pool;
use mempool::pool::PoolTrait;
use neuroxide::ops::add::Add;
use neuroxide::ops::mul::Mul;
use neuroxide::types::op_stub::OperationStub;
use neuroxide::types::tensor::SliceInfo;
use neuroxide::types::tensor::Tensor;
use neuroxide::types::tensor_element::TensorHandleExt;

extern crate neuroxide;

// TODO: Fix benchmarks
// TODO: Replace NDArray (Maybe)
// TODO: Create Union Graph for operations on CUDA
// TODO: Enable caching/saving
// TODO: Add more operations
fn main() {
    // let mut pool = Pool::new(CudaDevice::new());
    // let block1 = pool.malloc(1024).expect("Failed to allocate block1");
    // let block2 = pool.malloc(2048).expect("Failed to allocate block2");
    // pool.print();
    //
    // pool.free(&block1);
    // pool.print();
    //
    // pool.free(&block2);
    // pool.print();
    // // sleep for a while to see the output
    // let ten_millis = time::Duration::from_millis(5000);
    // std::thread::sleep(ten_millis);
    // let x = Tensor::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2usize, 3usize]);
    // let y = Tensor::new([1.0, 2.0, 3.0], [1usize, 3usize]);
    // let res = Tensor::cat(&x, &y, 0);
    // res.backward();
    // println!("Gradient dy/dx: {:?}", x.get_gradient());
    // println!("Gradient dy/dy: {:?}", y.get_gradient());
    //
    let x = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]); // shape [3]
    let y = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]); // shape [3]
    let z = Tensor::new(vec![7.0, 8.0, 9.0, 10.0], vec![4]); // shape [4]
    let w = Add::forward((&x, &y)); // [5,7,9]
    let u = Mul::forward((&w, &y)); // Error: shape mismatch
    let v = Tensor::cat(&u, &z, 0);
    v.backward();
    println!("Gradient dv/dx: {:?}", x.get_gradient());
    println!("Gradient dv/dy: {:?}", y.get_gradient());
    println!("Gradient dv/dz: {:?}", z.get_gradient());

    let x = Tensor::new([0, 1, 2, 3, 4, 5, 6, 7], [2usize, 4usize]);
    let y = Tensor::permute(&x, Box::new([1usize, 0usize]));
    println!("Permuted Tensor: {:?}", y);
    y.backward();
    println!("Gradient dy/dx: {:?}", x.get_gradient());

    let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    let shape = vec![4usize, 3usize, 1usize];
    let tensor = Tensor::new(data.clone(), shape.clone());
    println!("{:?}", tensor);
    tensor.lock().unwrap().print();

    // x.lock().unwrap().cuda();
    //
    // // (x * x)
    // let x2 = Mul::forward((&x, &x));
    // // (x * (x + x))
    // let x_plus_x = Add::forward((&x, &x));
    // let x_xplusx = Mul::forward((&x, &x_plus_x));
    // // (x*x + x)
    // let x2_plus_x = Add::forward((&x2, &x));
    // // y = (x*x + x) * (x * (x + x))
    // let y = Mul::forward((&x2_plus_x, &x_xplusx));
    //
    // // Trigger backward pass
    // y.backward();
    //
    // println!("Forward result y: {:?}", y);
    // println!("Gradient dy/dx: {:?}", x.get_gradient());
}
fn print_tensor<T: std::fmt::Display>(data: &[T], shape: &[usize]) {
    let ndim = shape.len();
    let mut stack = vec![0; ndim];
    for idx in 0..data.len() {
        let mut multi_idx = vec![0; ndim];
        let mut residual = idx;
        for i in (0..ndim).rev() {
            let dim = shape[i];
            multi_idx[i] = residual % dim;
            residual /= dim;
        }
        // Print opening brackets when a new slice along any axis starts
        for i in 0..ndim {
            if multi_idx[i] == 0 {
                if stack[i] == 0 {
                    stack[i] = 1;
                    print!("[");
                }
            }
        }

        // Print the value
        print!("{}", data[idx]);

        // Print closing brackets when a slice along any axis ends
        for i in (0..ndim).rev() {
            if multi_idx[i] + 1 == shape[i] {
                if stack[i] == 1 {
                    stack[i] = 0;
                    print!("]");
                }
            } else {
                print!(", ");
                break;
            }
        }
    }
}
