use core::time;

// use cuda::memory::cudadevice::CudaDevice;
use mempool::device_allocator::DeviceAllocator;
use mempool::pool::Pool;
use mempool::pool::PoolTrait;
use neuroxide::ops::add::Add;
use neuroxide::ops::matmul::Matmul;
use neuroxide::ops::mul::Mul;
use neuroxide::types::op_stub::OperationStub;
use neuroxide::types::tensor::SliceInfo;
use neuroxide::types::tensor::Tensor;
use neuroxide::types::tensor_element::TensorHandleExt;

extern crate neuroxide;

#[link(name = "openblas")]
unsafe extern "C" {
    fn sgemm_(
        transa: *const i8,
        transb: *const i8,
        m: *const i32,
        n: *const i32,
        k: *const i32,
        alpha: *const f32,
        a: *const f32,
        lda: *const i32,
        b: *const f32,
        ldb: *const i32,
        beta: *const f32,
        c: *mut f32,
        ldc: *const i32,
    );
}
fn sgemm_helper(
    m: usize,
    n: usize,
    k: usize,
    a_data: &[f32],
    b_data: &[f32],
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut c_data = vec![0f32; m * n];

    unsafe {
        // 'N' = no transpose
        let transa = b"T" as *const u8 as *const i8;
        let transb = b"T" as *const u8 as *const i8;

        sgemm_(
            transa,
            transb,
            &(n as i32),
            &(m as i32),
            &(k as i32),
            &alpha,
            a_data.as_ptr(),
            &(k as i32), // leading dimension of A
            b_data.as_ptr(),
            &(k as i32), // leading dimension of B
            &beta,
            c_data.as_mut_ptr(),
            &(m as i32), // leading dimension of C
        );
    }

    c_data
}

fn batched_sgemm(
    a_data: &[f32],
    b_data: &[f32],
    shape_a: &[usize], // [..., M, K]
    shape_b: &[usize], // [..., K, N]
) -> Vec<f32> {
    // Determine batch dims
    let batch_dims: Vec<usize> = if shape_a.len() <= 2 {
        vec![]
    } else {
        shape_a[..shape_a.len() - 2].to_vec()
    };
    let batch_size: usize = batch_dims.iter().product::<usize>().max(1);

    let m = shape_a[shape_a.len() - 2];
    let k = shape_a[shape_a.len() - 1];
    let n = shape_b[shape_b.len() - 1];

    let mut out_data = vec![0f32; batch_size * m * n];

    for batch_index in 0..batch_size {
        // Slice offsets
        let a_start = batch_index * m * k;
        let a_end = a_start + m * k;
        let b_start = batch_index * k * n;
        let b_end = b_start + k * n;
        let c_start = batch_index * m * n;
        let c_end = c_start + m * n;

        // Slices for this batch
        let a_slice = &a_data[a_start..a_end];
        let b_slice = &b_data[b_start..b_end];

        // Call your existing sgemm_helper
        let c_slice = sgemm_helper(m, n, k, a_slice, b_slice, 1.0, 0.0);

        // Copy result into output
        out_data[c_start..c_end].copy_from_slice(&c_slice);
    }

    out_data
}

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
    let a = Tensor::new(
        vec![
            1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ],
        vec![2, 1, 2, 4],
    );
    let b = Tensor::new(vec![5.0f32, 6.0, 7.0, 8.0], vec![4]);
    let res = Matmul::forward((&a, &b));
    res.lock().unwrap().print();
}
