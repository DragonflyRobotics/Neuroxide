use crate::ops::op_generic::{Ops, Operation};
use crate::types::device::Device;
use crate::types::tensor::Tensor;
use crate::types::t::TensorElement;
use crate::utils::node_uid::make_node_uid;
use cfg_if::cfg_if;


#[cfg(feature = "cuda")]
unsafe extern "C" {
pub fn cos_kernel(len: i32, a: *mut f32, c: *mut*mut f32) -> CudnnStatusT;
}

pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses

#[derive(Debug, Clone)]
pub struct CosOp;

impl<T> Operation<T> for CosOp
where
    T: TensorElement
{
    fn forward(inputs: &Vec<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 1);
        let result: Vec<T>; // = vec![T::default(); len as usize];

        cfg_if! {
            if #[cfg(feature = "cuda")] {
                let mut cuda_ptr: Option<*mut f32> = None;
            } else {
                let cuda_ptr: Option<*mut f32> = None;
            }
        }


        match inputs[0].device {
            Device::CPU => {
                result = inputs[0].data.iter().map(|x| x.cos()).collect();
            }
            Device::CUDA => {

                #[cfg(feature = "cuda")]
                unsafe {
                    let len = inputs[0].data.len() as i32;
                    let mut data: f32 = 0.0;
                    let mut ptr_to_data: *mut f32 = &mut data;
                    cos_kernel(len, inputs[0].cuda_ptr.unwrap(), &mut ptr_to_data);
                    cuda_ptr = Some(ptr_to_data);
                    let r = vec![0.0f32; len as usize];
                    result = r.iter().map(|&x| <T as num::NumCast>::from(x).unwrap()).collect();
                }
                #[cfg(not(feature = "cuda"))]
                {
                    panic!("CUDA feature not enabled");
                }
            }
        }


        //merge graphs
        let mut result_graph = inputs[0].op_chain.clone();

        let result_id = make_node_uid();
        result_graph.add_node(result_id);
        result_graph.add_edge(result_id, inputs[0].op_head, make_node_uid());

        let t = Tensor {
            id: result_id,
            data: result,
            shape: inputs[0].shape.clone(),
            device: inputs[0].device,
            op: Ops::CosEnum,
            requires_grad: inputs[0].requires_grad,
            op_chain: result_graph,
            op_head: result_id,
            dtype: inputs[0].dtype.clone(),
            cuda_ptr
        };



        let db = inputs[0].dtype.clone();
        db.write().unwrap().insert(t.clone());
        drop(db);
        t
    }

    fn backward(inputs: &Vec<&Tensor<T>>, _grad: Option<&Tensor<T>>) -> Tensor<T> {
        assert!(inputs.len() == 1);
        let mut grad_data = vec![T::default(); inputs[0].data.len()];
        for i in 0..inputs[0].data.len() {
            grad_data[i] = T::from(-1).unwrap() * inputs[0].data[i].sin();
        }
        Tensor {
            id: inputs[0].id,
            data: grad_data,
            shape: inputs[0].shape.clone(),
            device: Device::CPU,
            op: Ops::CosEnum,
            requires_grad: inputs[0].requires_grad,
            op_chain: inputs[0].op_chain.clone(),
            op_head: inputs[0].op_head,
            dtype: inputs[0].dtype.clone(),
            cuda_ptr: None // TODO: Fix this
        }
    }
}
