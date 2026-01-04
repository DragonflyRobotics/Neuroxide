use crate::{
    input::ToTensorInputs,
    op_stub::OperationStub,
    tensor::Tensor,
    tensor_element::{SharedTensor, TensorElement, TensorHandleExt},
};

#[derive(Debug)]
pub struct Cat<T> {
    pub input_tensors: Box<[SharedTensor<T>]>,
    pub axis: usize,
}

impl<T: TensorElement> OperationStub<T> for Cat<T> {
    fn forward<I: ToTensorInputs<T>>(inputs: I) -> SharedTensor<T> {
        todo!("Not allowed operation");
    }

    fn backward(&mut self, upstream: SharedTensor<T>) {
        // Backward implementation for Add operation
        println!("{:?}", upstream.get_shape());
        println!("{:?}", self.get_branches());
        let mut outer_dims = 1;
        let mut inner_dims = 1;
        for i in 0..self.get_branches()[0].get_shape().len() {
            if i != self.axis
                && self.get_branches()[0].get_shape()[i] != self.get_branches()[1].get_shape()[i]
            {
                panic!(
                    "Tensors must have the same shape except along the concatenation self.axis."
                );
            }
            if i < self.axis {
                outer_dims *= self.get_branches()[0].get_shape()[i];
            }
            if i > self.axis {
                inner_dims *= self.get_branches()[0].get_shape()[i];
            }
        }

        let self_slab_size = self.get_branches()[0].get_shape()[self.axis] * inner_dims;
        let other_slab_size = self.get_branches()[1].get_shape()[self.axis] * inner_dims;

        let mut final_shape = self.get_branches()[0].get_shape().clone();
        final_shape[self.axis] += self.get_branches()[1].get_shape()[self.axis];

        let mut a_vector: Vec<T> =
            Vec::with_capacity(self.get_branches()[0].get_shape().iter().product());
        unsafe {
            a_vector.set_len(self.get_branches()[0].get_shape().iter().product());
        }

        let mut b_vector: Vec<T> =
            Vec::with_capacity(self.get_branches()[1].get_shape().iter().product());
        unsafe {
            b_vector.set_len(self.get_branches()[1].get_shape().iter().product());
        }

        for i in 0..outer_dims {
            let mut starta = 0;
            let mut startb = 0;
            let mut residual = i;
            for d in 0..self.axis {
                let dim_size = self.get_branches()[0].get_shape()[d];
                let idx = residual % dim_size; // index along this dim
                residual /= dim_size; // update residual for next dim
                starta += idx * self.get_branches()[0].get_stride()[d];
                startb += idx * self.get_branches()[1].get_stride()[d];
            }

            let out_base = i * (self_slab_size + other_slab_size);
            a_vector[starta..starta + self_slab_size]
                .copy_from_slice(&upstream.values()[out_base..out_base + self_slab_size]);
            b_vector[startb..startb + other_slab_size].copy_from_slice(
                &upstream.values()
                    [out_base + self_slab_size..out_base + self_slab_size + other_slab_size],
            );
        }

        let a_grad = Tensor::new(a_vector, self.get_branches()[0].get_shape().clone());
        let b_grad = Tensor::new(b_vector, self.get_branches()[1].get_shape().clone());
        self.get_branches()[0].lock().unwrap().set_gradient(a_grad);
        self.get_branches()[1].lock().unwrap().set_gradient(b_grad);
        Self::recurse_backward(self.input_tensors[0].clone());
        Self::recurse_backward(self.input_tensors[1].clone());
    }

    fn get_branches(&self) -> Box<[SharedTensor<T>]> {
        self.input_tensors.clone()
    }
}
