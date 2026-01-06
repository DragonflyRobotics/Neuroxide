use neuroxide::ops::add::Add;
use neuroxide::ops::matmul::Matmul;
use neuroxide::ops::mul::Mul;
use neuroxide::types::op_stub::OperationStub;
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

    let a = Tensor::new(vec![1.0], vec![1]); // Scalar tensor
    let b = Tensor::new(
        (1..=12).map(|x| x as f32).collect::<Vec<f32>>(),
        vec![2, 3, 2],
    ); // [2,3,2]
    let c = Mul::forward((&a, &b)); // Broadcasting add
    c.lock().unwrap().print(); // [3,2]
    c.backward();
    a.get_gradient().unwrap().lock().unwrap().print();
    b.get_gradient().unwrap().lock().unwrap().print();
}
