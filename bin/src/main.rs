use std::time::Instant;

use neuroxide::ops::add::Add;
use neuroxide::ops::matmul::Matmul;
use neuroxide::ops::mul::Mul;
use neuroxide::ops::op::Operation;
use neuroxide::types::tensor::Tensor;
use neuroxide::types::tensor_element::TensorHandleExt;

// TODO: Fix benchmarks
// TODO: Replace NDArray (Maybe)
// TODO: Create Union Graph for operations on CUDA
// TODO: Enable caching/saving
// TODO: Add more operations
fn main() {
    // for _ in 0..100000 {
    //     // --- Step 1: Create base tensors ---
    //     let x = Tensor::new(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
    //     let y = Tensor::new(vec![10.0f32, 20.0, 30.0, 40.0, 50.0, 60.0], vec![2, 3]);
    //
    //     // --- Step 2: Basic arithmetic ---
    //     let z1 = Add::forward((&x, &y)); // elementwise add
    //     let z2 = Mul::forward((&x, &y)); // elementwise mul
    //
    //     // --- Step 3: Concatenate along axis 0 and 1 ---
    //     let cat0 = Tensor::cat(&z1, &z2, 0); // shape: [4, 3]
    //     let cat1 = Tensor::cat(&z1, &z2, 1); // shape: [2, 6]
    //
    //     // --- Step 4: Slice ---
    //     let slice0 = Tensor::slice(
    //         &cat0,
    //         &[
    //             SliceInfo::Range {
    //                 start: 1,
    //                 end: 3,
    //                 step: 1,
    //             },
    //             SliceInfo::All,
    //         ],
    //     ); // shape: [2, 3]
    //     let slice1 = Tensor::slice(
    //         &cat1,
    //         &[
    //             SliceInfo::All,
    //             SliceInfo::Range {
    //                 start: 2,
    //                 end: 5,
    //                 step: 1,
    //             },
    //         ],
    //     ); // shape: [2, 3]
    //
    //     // --- Step 5: View and reshape ---
    //     let view0 = Tensor::view(&slice0, vec![3, 2].into_boxed_slice()); // reshaped tensor
    //     let view1 = Tensor::view(&slice1, vec![3, 2].into_boxed_slice());
    //
    //     // --- Step 6: Unsqueeze and squeeze ---
    //     let unsq = Tensor::unsqueeze(&view0, 1); // shape: [3,1,2]
    //     let sq = Tensor::squeeze(&unsq, 1); // back to shape: [3,2]
    //
    //     // --- Step 7: Permute ---
    //     let perm = Tensor::permute(&sq, vec![1, 0].into_boxed_slice()); // shape: [2,3]
    //
    //     // --- Step 8: Combine with arithmetic again ---
    //     let shift = Tensor::permute(&view1, vec![1, 0].into_boxed_slice()); // shape: [2,3]
    //     let final_tensor = Add::forward((&perm, &shift)); // shapes must match [2,3]
    //     // final_tensor.lock().unwrap().print();
    //
    //     // --- Step 9: Backward pass ---
    //     final_tensor.backward(); // compute gradients through the entire chain
    //
    //     // --- Step 10: Print shapes and gradients ---
    //     // println!("x shape: {:?}", x.get_shape());
    //     // println!("y shape: {:?}", y.get_shape());
    //     // println!("final shape: {:?}", final_tensor.get_shape());
    //
    //     // x.get_gradient().unwrap().lock().unwrap().print();
    //     // y.get_gradient().unwrap().lock().unwrap().print();
    // }
    //
    // let a = Tensor::new(
    //     vec![
    //         1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
    //     ],
    //     vec![2, 1, 2, 4],
    // );
    let iters = 1000;
    let a = Tensor::new(vec![1.0f32; 1024 * 1024], vec![1024, 1024]);
    let b = Tensor::new(vec![2.0f32; 1024 * 1024], vec![1024, 1024]);

    let start = Instant::now();
    for i in 0..iters {
        let c = Add::forward((&a, &b));
        let step = start.elapsed();
        println!("Avg time per iter {}: {:?}", i + 1, step / (i + 1) as u32);
    }
    let elapsed = start.elapsed();
    println!(
        "[Simple Add] iters={} total={:?} per_iter={:?}",
        iters,
        elapsed,
        elapsed / iters as u32
    );

    // elementwise_reduction_bench(iters);
    // matmul_chain_bench(iters);
    // mixed_graph_bench(iters);
}

fn elementwise_reduction_bench(iters: usize) {
    let x = Tensor::new(vec![1.0f32; 1024 * 1024], vec![1024, 1024]);
    let y = Tensor::new(vec![2.0f32; 1024 * 1024], vec![1024, 1024]);

    // warmup
    {
        let z = Add::forward((&x, &y));
        let w = Mul::forward((&z, &x));
        let s = Tensor::sum(&w, 1);
        s.backward();
    }

    let start = Instant::now();

    for _ in 0..iters {
        let z1 = Add::forward((&x, &y));
        let z2 = Mul::forward((&z1, &x));
        let z3 = Add::forward((&z2, &z1));
        let z4 = Mul::forward((&z3, &z2));
        let out = Tensor::sum(&z4, 1); // reduction
        out.backward();
    }

    let elapsed = start.elapsed();
    println!(
        "[Elementwise+Reduce] iters={} total={:?} per_iter={:?}",
        iters,
        elapsed,
        elapsed / iters as u32
    );
}

fn matmul_chain_bench(iters: usize) {
    let a = Tensor::new(vec![1.0f32; 512 * 512], vec![512, 512]);
    let b = Tensor::new(vec![2.0f32; 512 * 512], vec![512, 512]);

    // warmup
    {
        let c = Matmul::forward((&a, &b));
        let d = Matmul::forward((&c, &a));
        d.backward();
    }

    let start = Instant::now();

    for _ in 0..iters {
        let c1 = Matmul::forward((&a, &b));
        let c2 = Matmul::forward((&c1, &a));
        let c3 = Matmul::forward((&c2, &b));
        let out = Tensor::sum(&c3, 0);
        out.backward();
        println!("Done an iteration");
    }

    let elapsed = start.elapsed();
    println!(
        "[MatMul Chain] iters={} total={:?} per_iter={:?}",
        iters,
        elapsed,
        elapsed / iters as u32
    );
}

fn mixed_graph_bench(iters: usize) {
    let x = Tensor::new(vec![1.0f32; 256 * 256], vec![256, 256]);
    let w = Tensor::new(vec![0.5f32; 256 * 256], vec![256, 256]);
    let b = Tensor::new(vec![0.1f32; 256], vec![256]);

    // warmup
    {
        let y = Mul::forward((&x, &w));
        let z = Tensor::sum(&y, 1);
        let o = Add::forward((&z, &b));
        o.backward();
    }

    let start = Instant::now();

    for _ in 0..iters {
        let y1 = Mul::forward((&x, &w));
        let y2 = Add::forward((&y1, &x));
        let y3 = Tensor::permute(&y2, vec![1, 0].into_boxed_slice());
        let y4 = Tensor::view(&y3, vec![256, 256].into_boxed_slice());
        let y5 = Mul::forward((&y4, &w));
        let z = Tensor::sum(&y5, 1);
        let out = Add::forward((&z, &b));
        out.backward();
    }

    let elapsed = start.elapsed();
    println!(
        "[Mixed Graph] iters={} total={:?} per_iter={:?}",
        iters,
        elapsed,
        elapsed / iters as u32
    );
}
