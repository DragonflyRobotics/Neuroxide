use crate::op::Operation;
use std::sync::{Arc, Mutex};
use types::{
    input::ToTensorInputs,
    op_stub::OperationStub,
    tensor::Tensor,
    tensor_element::{SharedTensor, TensorElement, TensorHandleExt},
};

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

/// Single GEMM call (row-major, no transpose)
fn sgemm_helper(
    m: usize,  // rows of A
    n: usize,  // cols of B
    k: usize,  // cols of A = rows of B
    a: &[f32], // flattened A [m*k]
    b: &[f32], // flattened B [k*n]
    alpha: f32,
    beta: f32,
) -> Vec<f32> {
    let mut c = vec![0f32; m * n];
    unsafe {
        sgemm_(
            b"N".as_ptr() as *const i8, // A not transposed
            b"N".as_ptr() as *const i8, // B not transposed
            &(n as i32),
            &(m as i32),
            &(k as i32),
            &alpha,
            b.as_ptr(),
            &(n as i32), // lda = number of columns in A
            a.as_ptr(),
            &(k as i32), // ldb = number of columns in B
            &beta,
            c.as_mut_ptr(),
            &(n as i32), // ldc = number of columns in C
        );
    }
    c
}

/// Batched GEMM for tensors with arbitrary batch dims
/// A shape: [..., M, K]
/// B shape: [..., K, N]
/// Returns flattened output: [..., M, N]
fn batched_sgemm(a: &[f32], b: &[f32], shape_a: &[usize], shape_b: &[usize]) -> Vec<f32> {
    assert!(shape_a.len() >= 2 && shape_b.len() >= 2);

    let m = shape_a[shape_a.len() - 2];
    let k_a = shape_a[shape_a.len() - 1];
    let k_b = shape_b[shape_b.len() - 2];
    let n = shape_b[shape_b.len() - 1];
    assert_eq!(k_a, k_b, "Inner dimensions must match!");

    // batch dimensions (everything except last 2 dims)
    let batch_dims = &shape_a[..shape_a.len() - 2];
    let batch_size = batch_dims.iter().product::<usize>().max(1);

    let mut out = vec![0f32; batch_size * m * n];

    for batch_idx in 0..batch_size {
        let a_start = batch_idx * m * k_a;
        let a_end = a_start + m * k_a;
        let b_start = batch_idx * k_a * n;
        let b_end = b_start + k_a * n;
        let c_start = batch_idx * m * n;
        let c_end = c_start + m * n;

        let a_slice = &a[a_start..a_end];
        let b_slice = &b[b_start..b_end];

        let c_slice = sgemm_helper(m, n, k_a, a_slice, b_slice, 1.0, 0.0);

        out[c_start..c_end].copy_from_slice(&c_slice);
    }
    out
}
#[derive(Debug)]
pub struct Matmul<T> {
    input_tensors: Box<[SharedTensor<T>]>,
}

impl<T: TensorElement> OperationStub<T> for Matmul<T> {
    fn forward<I: ToTensorInputs<T>>(inputs: I) -> SharedTensor<T> {
        let inputs = inputs.into_inputs();
        if inputs.len() != 2 {
            panic!("Add operation requires exactly two input tensors.");
        }
        let (c, d) = Tensor::broadcast_matmul(&inputs[0], &inputs[1]);
        let c_data: Vec<f32> = c.values().iter().map(|x| x.to_f32().unwrap()).collect();
        let d_data: Vec<f32> = d.values().iter().map(|x| x.to_f32().unwrap()).collect();
        let result_data: Vec<T> = batched_sgemm(&c_data, &d_data, &c.get_shape(), &d.get_shape())
            .into_iter()
            .map(|x| T::from(x).unwrap())
            .collect();
        println!("{:?}", result_data);
        let mut result_shape = c.get_shape();
        result_shape[result_shape.len() - 1] = d.get_shape()[d.get_shape().len() - 1];
        let result_tensor = Tensor::new(result_data, result_shape);
        let matmul = Matmul {
            input_tensors: inputs.clone(),
        };
        result_tensor
            .lock()
            .unwrap()
            .set_op(Arc::new(Mutex::new(matmul)));
        result_tensor
    }

    fn backward(&mut self, upstream: SharedTensor<T>) {
        todo!()
    }

    fn get_branches(&self) -> Box<[SharedTensor<T>]> {
        self.input_tensors.clone()
    }
}
