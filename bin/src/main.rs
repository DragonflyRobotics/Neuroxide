use core::time;

// use cuda::memory::cudadevice::CudaDevice;
use mempool::device_allocator::DeviceAllocator;
use mempool::pool::Pool;
use mempool::pool::PoolTrait;
use neuroxide::ops::add::Add;
use neuroxide::ops::mul::Mul;
use neuroxide::types::op_stub::OperationStub;
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
    let x = Tensor::new([1.0, 2.0, 3.0, 4.0], [2usize, 2usize]);
    x.lock().unwrap().cuda();

    // (x * x)
    let x2 = Mul::forward((&x, &x));
    // (x * (x + x))
    let x_plus_x = Add::forward((&x, &x));
    let x_xplusx = Mul::forward((&x, &x_plus_x));
    // (x*x + x)
    let x2_plus_x = Add::forward((&x2, &x));
    // y = (x*x + x) * (x * (x + x))
    let y = Mul::forward((&x2_plus_x, &x_xplusx));

    // Trigger backward pass
    y.backward();

    println!("Forward result y: {:?}", y);
    println!("Gradient dy/dx: {:?}", x.get_gradient());
}
