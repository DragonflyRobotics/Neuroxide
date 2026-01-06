use std::{
    process::id,
    sync::{Arc, Mutex},
};

use crate::{
    broadcast::{Broadcast, BroadcastParams},
    cat::Cat,
    device::Device,
    input::{ToShapeInputs, ToTensorValueInputs},
    op_stub::OperationStub,
    permute::Permute,
    slice::Slice,
    squeeze::Squeeze,
    tensor_data::TensorData,
    tensor_element::{SharedTensor, TensorElement, TensorHandleExt},
    unsqueeze::Unsqueeze,
    view::View,
};

pub struct Tensor<T> {
    data: TensorData<T>,
    requires_grad: bool,
    gradient: Option<SharedTensor<T>>,
    op: Option<Arc<Mutex<dyn OperationStub<T>>>>,
}

#[derive(Debug)]
pub enum SliceInfo {
    All, // take full axis
    Range {
        start: usize,
        end: usize,
        step: usize,
    }, // slice with step
}

#[derive(Debug)]
pub(crate) struct ParsedSlice {
    pub(crate) start: usize,
    pub(crate) end: usize,
    pub(crate) step: usize,
    pub(crate) length: usize, // length along this axis in the output tensor
}

impl<T: TensorElement> Tensor<T> {
    pub fn new<I: ToShapeInputs, V: ToTensorValueInputs<T>>(data: V, shape: I) -> SharedTensor<T> {
        let data = data.into_values();
        let shape = shape.into_shape();
        let device = Device::CPU;
        let tensor_data = TensorData::new(data, shape, device).unwrap();
        Arc::new(Mutex::new(Tensor {
            data: tensor_data,
            requires_grad: false,
            gradient: None,
            op: None,
        }))
    }

    pub fn ones<I: ToShapeInputs>(shape: I) -> SharedTensor<T> {
        let shape = shape.into_shape();
        let device = Device::CPU;
        let tensor_data = TensorData::new(
            vec![T::from(1).unwrap(); shape.clone().into_iter().product()],
            shape,
            device,
        )
        .unwrap();
        Arc::new(Mutex::new(Tensor {
            data: tensor_data,
            requires_grad: false,
            gradient: None,
            op: None,
        }))
    }

    pub fn set_op(&mut self, op: Arc<Mutex<dyn OperationStub<T>>>) {
        self.op = Some(op);
    }

    pub fn get_values(&self) -> &Vec<T> {
        &self.data.values
    }

    pub fn get_shape(&self) -> Box<[usize]> {
        self.data.shape.clone()
    }

    pub fn get_op(&self) -> Option<Arc<Mutex<dyn OperationStub<T>>>> {
        self.op.clone()
    }

    pub fn backward(&mut self) {
        self.gradient = Some(Tensor::ones(self.get_shape().clone()));
        let binding = self.op.clone().unwrap();
        let mut op = binding.lock().unwrap();
        // Placeholder for future upstream gradients
        op.backward(self.gradient.clone().unwrap());
    }

    pub fn set_gradient(&mut self, grad: SharedTensor<T>) {
        self.gradient = Some(grad);
    }

    pub fn get_gradient(&self) -> Option<SharedTensor<T>> {
        self.gradient.clone()
    }

    pub fn get_stride(&self) -> Box<[usize]> {
        let mut stride = vec![1; self.data.shape.len()];
        for i in (0..self.data.shape.len() - 1).rev() {
            stride[i] = stride[i + 1] * self.data.shape[i + 1];
        }
        stride.into_boxed_slice()
    }

    pub fn cat(one: &SharedTensor<T>, other: &SharedTensor<T>, axis: usize) -> SharedTensor<T> {
        if one.get_shape().len() != other.get_shape().len() {
            panic!("Tensors must have the same number of dimensions to concatenate.");
        }

        let mut outer_dims = 1;
        let mut inner_dims = 1;
        for i in 0..one.get_shape().len() {
            if i != axis && one.get_shape()[i] != other.get_shape()[i] {
                panic!("Tensors must have the same shape except along the concatenation axis.");
            }
            if i < axis {
                outer_dims *= one.get_shape()[i];
            }
            if i > axis {
                inner_dims *= one.get_shape()[i];
            }
        }

        let self_slab_size = one.get_shape()[axis] * inner_dims;
        let other_slab_size = other.get_shape()[axis] * inner_dims;

        let mut final_shape = one.get_shape().clone();
        final_shape[axis] += other.get_shape()[axis];

        let mut final_vector: Vec<T> = Vec::with_capacity(final_shape.iter().product());
        unsafe {
            final_vector.set_len(final_shape.iter().product());
        }

        for i in 0..outer_dims {
            let mut starta = 0;
            let mut startb = 0;
            let mut residual = i;
            for d in 0..axis {
                let dim_size = one.get_shape()[d];
                let idx = residual % dim_size; // index along this dim
                residual /= dim_size; // update residual for next dim
                starta += idx * one.get_stride()[d];
                startb += idx * other.get_stride()[d];
            }

            let out_base = i * (self_slab_size + other_slab_size);
            final_vector[out_base..out_base + self_slab_size]
                .copy_from_slice(&one.values()[starta..starta + self_slab_size]);

            final_vector[out_base + self_slab_size..out_base + self_slab_size + other_slab_size]
                .copy_from_slice(&other.values()[startb..startb + other_slab_size]);
        }

        // TODO: Device handling
        let data = TensorData::new(final_vector.clone(), final_shape.clone(), Device::CPU).unwrap();
        let op: Option<Arc<Mutex<dyn OperationStub<T>>>> = Some(Arc::new(Mutex::new(Cat {
            input_tensors: Box::new([one.clone(), other.clone()]),
            axis,
        })));
        Arc::new(Mutex::new(Tensor {
            data,
            requires_grad: one.lock().unwrap().requires_grad || other.lock().unwrap().requires_grad,
            gradient: None,
            op,
        }))
    }

    pub fn sum(one: &SharedTensor<T>, axis: usize) -> SharedTensor<T> {
        let shape = one.get_shape();
        let ndim = shape.len();
        assert!(axis < ndim);

        let mut outer = 1usize;
        let mut inner = 1usize;

        for i in 0..axis {
            outer *= shape[i];
        }
        for i in axis + 1..ndim {
            inner *= shape[i];
        }

        let reduce = shape[axis];

        // Output shape (keepdim = true)
        let mut out_shape = shape.clone();
        out_shape[axis] = 1;

        let out_len = outer * inner;
        let mut out = vec![T::from(0).unwrap(); out_len];

        let input = one.values();

        // Core reduction
        for o in 0..outer {
            let in_base = o * reduce * inner;
            let out_base = o * inner;

            for i in 0..inner {
                let mut acc = T::from(0).unwrap();

                for r in 0..reduce {
                    acc = acc + input[in_base + r * inner + i];
                }

                out[out_base + i] = acc;
            }
        }

        let data = TensorData::new(out, out_shape, Device::CPU).unwrap();

        let op: Option<Arc<Mutex<dyn OperationStub<T>>>> =
            Some(Arc::new(Mutex::new(crate::axis_sum::AxisSum {
                input_tensors: Box::new([one.clone()]),
                axis,
            })));

        Arc::new(Mutex::new(Tensor {
            data,
            requires_grad: one.lock().unwrap().requires_grad,
            gradient: None,
            op,
        }))
    }

    pub(crate) fn linear_to_multi(idx: usize, shape: &[usize]) -> Vec<usize> {
        let mut indices = vec![0; shape.len()];
        let mut residual = idx;
        for i in (0..shape.len()).rev() {
            indices[i] = residual % shape[i];
            residual /= shape[i];
        }
        indices
    }

    pub fn slice(one: &SharedTensor<T>, ranges: &[SliceInfo]) -> SharedTensor<T> {
        let mut output_shape = one.get_shape().clone();
        let parsed_slice = ranges
            .iter()
            .enumerate()
            .map(|(i, range)| match range {
                SliceInfo::All => ParsedSlice {
                    start: 0,
                    end: one.get_shape()[i],
                    step: 1,
                    length: one.get_shape()[i],
                },
                SliceInfo::Range { start, end, step } => {
                    let dim_size = one.get_shape()[i];
                    if *end > dim_size || *start >= *end {
                        panic!("Slice indices are out of bounds.");
                    }
                    let slice_size = ((*end - *start) + step - 1) / step;
                    output_shape[i] = slice_size;
                    ParsedSlice {
                        start: *start,
                        end: *end,
                        step: *step,
                        length: slice_size,
                    }
                }
            })
            .collect::<Vec<ParsedSlice>>();

        let output_numel: usize = output_shape.iter().product();
        let mut output_values: Vec<T> = Vec::with_capacity(output_numel);
        for out_linear in 0..output_numel {
            let out_multi = Tensor::<T>::linear_to_multi(out_linear, &output_shape);
            let mut in_multi = vec![0; out_multi.len()];
            for i in 0..out_multi.len() {
                let ps = &parsed_slice[i];
                in_multi[i] = ps.start + out_multi[i] * ps.step;
            }
            let in_linear: usize = in_multi
                .iter()
                .enumerate()
                .map(|(dim, &idx)| idx * one.get_stride()[dim])
                .sum();
            output_values.push(one.values()[in_linear]);
        }

        let tensor_data =
            TensorData::new(output_values.clone(), output_shape.clone(), Device::CPU).unwrap();
        let op: Option<Arc<Mutex<dyn OperationStub<T>>>> = Some(Arc::new(Mutex::new(Slice {
            input_tensors: Box::new([one.clone()]),
            parsed_slices: parsed_slice,
        })));

        Arc::new(Mutex::new(Tensor {
            data: tensor_data,
            requires_grad: one.lock().unwrap().requires_grad,
            gradient: None,
            op,
        }))
    }

    pub fn unsqueeze(one: &SharedTensor<T>, axis: usize) -> SharedTensor<T> {
        let mut new_shape = one.get_shape().to_vec();
        new_shape.insert(axis, 1);
        // TODO: Device handling
        let tensor_data = TensorData::new(
            one.values().clone(),
            new_shape.into_boxed_slice(),
            Device::CPU,
        )
        .unwrap();
        let op: Option<Arc<Mutex<dyn OperationStub<T>>>> = Some(Arc::new(Mutex::new(Unsqueeze {
            input_tensors: Box::new([one.clone()]),
            axis,
        })));
        Arc::new(Mutex::new(Tensor {
            data: tensor_data,
            requires_grad: one.lock().unwrap().requires_grad,
            gradient: None,
            op,
        }))
    }

    pub fn squeeze(one: &SharedTensor<T>, axis: usize) -> SharedTensor<T> {
        let mut new_shape = one.get_shape().to_vec();
        assert!(
            new_shape[axis] == 1,
            "Cannot squeeze axis {} with size {}",
            axis,
            new_shape[axis]
        );
        new_shape.remove(axis);
        // TODO: Device handling
        let tensor_data = TensorData::new(
            one.values().clone(),
            new_shape.into_boxed_slice(),
            Device::CPU,
        )
        .unwrap();
        let op: Option<Arc<Mutex<dyn OperationStub<T>>>> = Some(Arc::new(Mutex::new(Squeeze {
            input_tensors: Box::new([one.clone()]),
            axis,
        })));
        Arc::new(Mutex::new(Tensor {
            data: tensor_data,
            requires_grad: one.lock().unwrap().requires_grad,
            gradient: None,
            op,
        }))
    }

    pub fn view(one: &SharedTensor<T>, shape: Box<[usize]>) -> SharedTensor<T> {
        let numel: usize = one.get_shape().iter().product();
        let new_numel: usize = shape.iter().product();
        assert!(
            numel == new_numel,
            "Cannot view tensor of numel {} as shape with numel {}",
            numel,
            new_numel
        );
        // TODO: Device handling
        let tensor_data =
            TensorData::new(one.values().clone(), shape.clone(), Device::CPU).unwrap();

        let op: Option<Arc<Mutex<dyn OperationStub<T>>>> = Some(Arc::new(Mutex::new(View {
            input_tensors: Box::new([one.clone()]),
            shape,
        })));

        Arc::new(Mutex::new(Tensor {
            data: tensor_data,
            requires_grad: one.lock().unwrap().requires_grad,
            gradient: None,
            op,
        }))
    }

    pub fn permute(one: &SharedTensor<T>, reordered_axis: Box<[usize]>) -> SharedTensor<T> {
        assert!(
            reordered_axis.len() == one.get_shape().len(),
            "Reordered axis length must match tensor dimensions."
        );
        let mut new_shape = vec![0; one.get_shape().len()];
        let mut new_stride = vec![0; one.get_shape().len()];
        for (i, &axis) in reordered_axis.iter().enumerate() {
            assert!(
                axis < one.get_shape().len(),
                "Reordered axis index out of bounds."
            );
            new_shape[i] = one.get_shape()[axis];
            new_stride[i] = one.get_stride()[axis];
        }
        assert!(
            new_shape.iter().product::<usize>() == one.get_shape().iter().product::<usize>(),
            "Total number of elements must remain the same after permutation."
        );

        let mut output_values: Vec<T> = vec![T::from(0).unwrap(); new_shape.iter().product()];
        for out_linear in 0..output_values.len() {
            let out_multi = Tensor::<T>::linear_to_multi(out_linear, &new_shape);
            let in_linear: usize = out_multi
                .iter()
                .enumerate()
                .map(|(dim, &idx)| idx * new_stride[dim])
                .sum();
            output_values[out_linear] = one.values()[in_linear];
        }

        // TODO: Device handling
        let tensor_data =
            TensorData::new(output_values, new_shape.into_boxed_slice(), Device::CPU).unwrap();

        let op: Option<Arc<Mutex<dyn OperationStub<T>>>> = Some(Arc::new(Mutex::new(Permute {
            input_tensors: Box::new([one.clone()]),
            reordered_axis,
        })));

        Arc::new(Mutex::new(Tensor {
            data: tensor_data,
            requires_grad: one.lock().unwrap().requires_grad,
            gradient: None,
            op,
        }))
    }

    pub fn broadcast_linear(
        one: &SharedTensor<T>,
        other: &SharedTensor<T>,
    ) -> (SharedTensor<T>, SharedTensor<T>) {
        let mut arg1_params = BroadcastParams {
            axes_to_sum: vec![],
            axes_to_remove: vec![],
        };
        let mut arg2_params = BroadcastParams {
            axes_to_sum: vec![],
            axes_to_remove: vec![],
        };
        let mut shape1 = one.get_shape().to_vec();
        let mut shape2 = other.get_shape().to_vec();
        let mut one_result = one.clone();
        let mut other_result = other.clone();

        for _ in 0..shape1.len() {
            arg1_params.axes_to_remove.push(false);
            arg1_params.axes_to_sum.push(false);
        }
        for _ in 0..shape2.len() {
            arg2_params.axes_to_remove.push(false);
            arg2_params.axes_to_sum.push(false);
        }

        // fill with zeros from left to right
        let diff = shape1.len() as i32 - shape2.len() as i32;
        if diff > 0 {
            for _ in 0..diff {
                shape2.insert(0, 1);
                arg2_params.axes_to_remove.insert(0, true);
                arg2_params.axes_to_sum.insert(0, false);
                other_result = Tensor::unsqueeze(&other_result, 0);
            }
        } else {
            for _ in 0..-diff {
                shape1.insert(0, 1);
                arg1_params.axes_to_remove.insert(0, true);
                arg1_params.axes_to_sum.insert(0, false);
                one_result = Tensor::unsqueeze(&one_result, 0);
            }
        }

        for index in 0..shape1.len() {
            if shape1[index] != shape2[index] {
                if shape1[index] == 1 {
                    shape1[index] = shape2[index];
                    arg1_params.axes_to_sum[index] = true;
                    let base = one_result.clone();
                    for _ in 0..(shape2[index]) - 1 {
                        one_result = Tensor::cat(&one_result, &base, index);
                    }
                } else if shape2[index] == 1 {
                    shape2[index] = shape1[index];
                    arg2_params.axes_to_sum[index] = true;
                    let base = other_result.clone();
                    for _ in 0..(shape1[index]) - 1 {
                        other_result = Tensor::cat(&other_result, &base, index);
                    }
                } else {
                    panic!(
                        "Shapes are not broadcastable: {:?} and {:?}.",
                        shape1, shape2
                    );
                }
            }
        }

        let one_op = Broadcast {
            input_tensors: Box::new([one.clone()]),
            arg_params: arg1_params.clone(),
        };

        let other_op = Broadcast {
            input_tensors: Box::new([other.clone()]),
            arg_params: arg2_params.clone(),
        };

        one_result = Arc::new(Mutex::new(Tensor {
            data: TensorData {
                device: Device::CPU,
                shape: one_result.get_shape(),
                values: one_result.values(),
                cuda_ptr: None,
            },
            requires_grad: false,
            gradient: None,
            op: Some(Arc::new(Mutex::new(one_op))),
        }));

        other_result = Arc::new(Mutex::new(Tensor {
            data: TensorData {
                device: Device::CPU,
                shape: other_result.get_shape(),
                values: other_result.values(),
                cuda_ptr: None,
            },
            requires_grad: false,
            gradient: None,
            op: Some(Arc::new(Mutex::new(other_op))),
        }));
        (one_result, other_result)
    }

    fn mat_dim_broad(
        one: &SharedTensor<T>,
        other: &SharedTensor<T>,
    ) -> (
        SharedTensor<T>,
        SharedTensor<T>,
        BroadcastParams,
        BroadcastParams,
    ) {
        let mut arg1_params = BroadcastParams {
            axes_to_sum: vec![],
            axes_to_remove: vec![],
        };
        let mut arg2_params = BroadcastParams {
            axes_to_sum: vec![],
            axes_to_remove: vec![],
        };
        let mut one_result = one.clone();
        let mut other_result = other.clone();
        let mut shape1 = one_result.get_shape().to_vec();
        let mut shape2 = other_result.get_shape().to_vec();
        for _ in 0..(shape1.len() as i32 - 2).max(0) {
            shape1.remove(0);
        }
        for _ in 0..(shape2.len() as i32 - 2).max(0) {
            shape2.remove(0);
        }
        let rank1 = shape1.len();
        let rank2 = shape2.len();
        for _ in 0..rank1 {
            arg1_params.axes_to_sum.push(false);
            arg1_params.axes_to_remove.push(false);
        }
        for _ in 0..rank2 {
            arg2_params.axes_to_sum.push(false);
            arg2_params.axes_to_remove.push(false);
        }
        if rank1 == 1 {
            if rank2 == 1 {
                // [z] X [z] = [1] -> []
                if shape1[0] == shape2[0] {
                    one_result = Tensor::unsqueeze(&one_result, shape1.len() - 1);
                    other_result = Tensor::unsqueeze(&other_result, shape2.len());
                    arg1_params.axes_to_remove.insert(0, true);
                    arg1_params.axes_to_sum.insert(0, false);
                    arg2_params.axes_to_remove.push(true);
                    arg2_params.axes_to_sum.push(false);
                    return (one_result, other_result, arg1_params, arg2_params);
                } else {
                    panic!(
                        "Shapes are not aligned for matmul: {:?} and {:?}.",
                        shape1, shape2
                    );
                }
            } else if rank2 == 2 {
                // [z] X [z, y] = [1, y] -> [y]
                if shape1[0] == shape2[0] {
                    arg1_params.axes_to_remove.insert(0, true);
                    arg1_params.axes_to_sum.insert(0, false);
                    one_result = Tensor::unsqueeze(&one_result, shape1.len() - 1);
                    return (one_result, other_result, arg1_params, arg2_params);
                } else {
                    panic!(
                        "Shapes are not aligned for matmul: {:?} and {:?}.",
                        shape1, shape2
                    );
                }
            }
        } else if rank1 == 2 {
            if rank2 == 1 {
                // [y, z] X [z] = [y, 1] -> [y]
                if shape1[1] == shape2[0] {
                    arg2_params.axes_to_remove.push(true);
                    arg2_params.axes_to_sum.push(false);
                    other_result = Tensor::unsqueeze(&other_result, shape2.len());
                    return (one_result, other_result, arg1_params, arg2_params);
                } else {
                    panic!(
                        "Shapes are not aligned for matmul: {:?} and {:?}.",
                        shape1, shape2
                    );
                }
            } else if rank2 == 2 {
                // [y, z] X [z, y] = [y, y] -> []
                if shape1[1] == shape2[0] {
                    return (one_result, other_result, arg1_params, arg2_params);
                } else {
                    panic!(
                        "Shapes are not aligned for matmul: {:?} and {:?}.",
                        shape1, shape2
                    );
                }
            }
        }
        panic!(
            "Shapes are not aligned for matmul: {:?} and {:?}.",
            shape1, shape2
        );
    }

    pub fn broadcast_matmul(
        one: &SharedTensor<T>,
        other: &SharedTensor<T>,
    ) -> (SharedTensor<T>, SharedTensor<T>) {
        let (mut one_result, mut other_result, mat1_params, mat2_params) =
            Tensor::mat_dim_broad(&one, &other);
        let (mut shape1, mut shape2) = (
            one_result.get_shape().to_vec(),
            other_result.get_shape().to_vec(),
        );
        if shape1.len() <= 2 && shape2.len() <= 2 {
            return (one_result, other_result);
        }

        let mut arg1_params = BroadcastParams {
            axes_to_sum: vec![],
            axes_to_remove: vec![],
        };
        let mut arg2_params = BroadcastParams {
            axes_to_sum: vec![],
            axes_to_remove: vec![],
        };
        for _ in 0..shape1.len() - 2 {
            arg1_params.axes_to_sum.push(false);
            arg1_params.axes_to_remove.push(false);
        }
        arg1_params
            .axes_to_sum
            .push(mat1_params.axes_to_sum[mat1_params.axes_to_sum.len() - 2]);
        arg1_params
            .axes_to_sum
            .push(mat1_params.axes_to_sum[mat1_params.axes_to_sum.len() - 1]);
        arg1_params
            .axes_to_remove
            .push(mat1_params.axes_to_remove[mat1_params.axes_to_remove.len() - 2]);
        arg1_params
            .axes_to_remove
            .push(mat1_params.axes_to_remove[mat1_params.axes_to_remove.len() - 1]);
        for _ in 0..shape2.len() - 2 {
            arg2_params.axes_to_sum.push(false);
            arg2_params.axes_to_remove.push(false);
        }
        arg2_params
            .axes_to_sum
            .push(mat2_params.axes_to_sum[mat2_params.axes_to_sum.len() - 2]);
        arg2_params
            .axes_to_sum
            .push(mat2_params.axes_to_sum[mat2_params.axes_to_sum.len() - 1]);
        arg2_params
            .axes_to_remove
            .push(mat2_params.axes_to_remove[mat2_params.axes_to_remove.len() - 2]);
        arg2_params
            .axes_to_remove
            .push(mat2_params.axes_to_remove[mat2_params.axes_to_remove.len() - 1]);

        if shape1.len() > shape2.len() {
            let diff = shape1.len() - shape2.len();
            for _ in 0..diff {
                shape2.insert(0, 1);
                arg2_params.axes_to_remove.insert(0, true);
                arg2_params.axes_to_sum.insert(0, false);
                other_result = Tensor::unsqueeze(&other_result, 0);
            }
        } else if shape2.len() > shape1.len() {
            let diff = shape2.len() - shape1.len();
            for _ in 0..diff {
                shape1.insert(0, 1);
                arg1_params.axes_to_remove.insert(0, true);
                arg1_params.axes_to_sum.insert(0, false);
                one_result = Tensor::unsqueeze(&one_result, 0);
            }
        }
        assert!(shape1.len() == shape2.len());
        for index in 0..shape1.len() - 2 {
            if shape1[index] != shape2[index] {
                if shape1[index] == 1 {
                    shape1[index] = shape2[index];
                    arg1_params.axes_to_sum[index] = true;
                    let base = one_result.clone();
                    for _ in 0..(shape2[index]) - 1 {
                        one_result = Tensor::cat(&one_result, &base, index);
                    }
                } else if shape2[index] == 1 {
                    shape2[index] = shape1[index];
                    arg2_params.axes_to_sum[index] = true;
                    let base = other_result.clone();
                    for _ in 0..(shape1[index]) - 1 {
                        other_result = Tensor::cat(&other_result, &base, index);
                    }
                } else {
                    panic!(
                        "Shapes are not broadcastable for matmul: {:?} and {:?}.",
                        shape1, shape2
                    );
                }
            }
        }
        let one_op = Broadcast {
            input_tensors: Box::new([one.clone()]),
            arg_params: arg1_params.clone(),
        };
        let other_op = Broadcast {
            input_tensors: Box::new([other.clone()]),
            arg_params: arg2_params.clone(),
        };

        one_result = Arc::new(Mutex::new(Tensor {
            data: TensorData {
                device: Device::CPU,
                shape: one_result.get_shape(),
                values: one_result.values(),
                cuda_ptr: None,
            },
            requires_grad: false,
            gradient: None,
            op: Some(Arc::new(Mutex::new(one_op))),
        }));

        other_result = Arc::new(Mutex::new(Tensor {
            data: TensorData {
                device: Device::CPU,
                shape: other_result.get_shape(),
                values: other_result.values(),
                cuda_ptr: None,
            },
            requires_grad: false,
            gradient: None,
            op: Some(Arc::new(Mutex::new(other_op))),
        }));

        (one_result, other_result)
    }

    pub fn print(&self) {
        // handling
        print!("tensor(");
        let ndim = self.data.shape.len();
        let mut stack = vec![0; ndim];
        for idx in 0..self.data.values.len() {
            let mut multi_idx = vec![0; ndim];
            let mut residual = idx;
            for i in (0..ndim).rev() {
                let dim = self.data.shape[i];
                multi_idx[i] = residual % dim;
                residual /= dim;
            }
            // Print opening brackets when a new slice along any axis starts
            for i in 0..ndim {
                if multi_idx[i] == 0 && stack[i] == 0 {
                    stack[i] = 1;
                    print!("[");
                }
            }

            // Print the value
            print!("{}", self.data.values[idx]);

            // Print closing brackets when a slice along any axis ends
            for i in (0..ndim).rev() {
                if multi_idx[i] + 1 == self.data.shape[i] {
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
        println!(
            ", shape={:?}, device={:?})",
            self.data.shape, self.data.device
        );
    }
}

impl<T> std::fmt::Debug for Tensor<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.data.values)
            .field("shape", &self.data.shape)
            .field("requires_grad", &self.requires_grad)
            .field("op", &self.op.is_some())
            .finish()
    }
}
