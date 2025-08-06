use std::{cmp::max, collections::HashMap, sync::{Arc, RwLock}, vec};

use crate::{ops::{add::AddOp, cos::CosOp, div::DivOp, ln::LnOp, matmul::MatMulOp, mul::MulOp, op_generic::{Operation, Ops}, pow::PowOp, sin::SinOp, sub::SubOp}, types::{device::Device, tensordb::DTypes}, utils::types::print_type_of};
use ndarray::{ArrayD, IxDyn};
use num::abs;
use petgraph::{algo, prelude::GraphMap, Directed, Direction::Outgoing};
use crate::utils::node_uid::make_node_uid;
use rand::Rng;
use cfg_if::cfg_if;

use super::{tensordb::{assert_types, TensorDB}, t::TensorElement};


#[cfg(feature = "cuda")]
unsafe extern "C" {
fn toCuda(size: i32, data: *mut f32) -> *mut f32;
fn checkkData(len: i32, ptr: *mut f32) -> i32;
fn toCpu(size: i32, ptr: *mut f32) -> *mut f32;
fn destroyPool();
fn createPoolMax();
pub fn reduce(d: i32, m: i32, n: i32, a: *mut f32, c: *mut*mut f32) -> CudnnStatusT;
}
pub type CudnnStatusT = i32; // usually cuDNN uses enums as return statuses

#[derive(Clone)]
pub struct Tensor<T> {
    pub id: i32,
    pub data: Vec<T>,
    pub shape: Vec<usize>,
    pub device: Device,
    pub op: Ops,
    pub requires_grad: bool,
    pub op_chain: GraphMap<i32, i32, Directed>,
    pub op_head: i32,
    pub dtype: Arc<RwLock<TensorDB<T>>>,
    pub cuda_ptr: Option<*mut f32>
}

impl<T> Tensor<T> 
where
    T: TensorElement
{
    pub fn new(db: &Arc<RwLock<TensorDB<T>>>, data: Vec<T>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor<T> {
        assert_types(db.read().unwrap().get_dtype(), data[0]);
        cfg_if! {
            if #[cfg(feature = "cuda")] {
                let mut cuda_ptr: Option<*mut f32> = None;
            } else {
                let cuda_ptr: Option<*mut f32> = None;
            }
        }
        if device == Device::CUDA {
            assert!(db.read().unwrap().get_dtype() == DTypes::F32, "CUDA only supports f32");
            #[cfg(feature = "cuda")]
            {
                let mut a: Vec<f32> = data.iter().map(|&x| <f32 as num::NumCast>::from(x).unwrap()).collect();
                cuda_ptr = Some(unsafe { toCuda(a.len() as i32, a.as_mut_ptr() as *mut f32) });
            }
        }
        let mut graph = GraphMap::new();
        let id = make_node_uid();
        graph.add_node(id);
        let t = Tensor {
            id,
            data,
            shape,
            device,
            op: Ops::TensorEnum,
            requires_grad,
            op_chain: graph,
            op_head: id,
            dtype: db.clone(),
            cuda_ptr: cuda_ptr
        };
        db.write().unwrap().insert(t.clone());
        t
    }

    pub fn cpu(&mut self) {
        if self.device == Device::CPU {
            return;
        }
        #[cfg(feature = "cuda")]
        unsafe {
            let len = self.shape.iter().product::<usize>() as i32;
            let h_A = toCpu(len, self.cuda_ptr.unwrap());
            let mut a = vec![0.0; len as usize];
            std::ptr::copy_nonoverlapping(h_A, a.as_mut_ptr(), len as usize);
            let a: Vec<T> = a.iter().map(|&x| <T as num::NumCast>::from(x).unwrap()).collect();
            self.data = a;
            self.cuda_ptr = None;
            self.device = Device::CPU;
        }
    }

    pub fn cuda(&mut self) {
        #[cfg(feature = "cuda")]
        unsafe {
            let mut a: Vec<f32> = self.data.iter().map(|&x| <f32 as num::NumCast>::from(x).unwrap()).collect();
            self.cuda_ptr = Some(toCuda(a.len() as i32, a.as_mut_ptr() as *mut f32));
            self.device = Device::CUDA;
        }
    }

    pub fn new_ones(db: &Arc<RwLock<TensorDB<T>>>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor<T> {
        let data = vec![T::from(1).unwrap(); shape.iter().product()];
        Tensor::new(db, data, shape, device, requires_grad)
    }

    pub fn new_zeros(db: &Arc<RwLock<TensorDB<T>>>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor<T> {
        let data = vec![T::from(0).unwrap(); shape.iter().product()];
        Tensor::new(db, data, shape, device, requires_grad)
    }

    pub fn new_uniform(db: &Arc<RwLock<TensorDB<T>>>, shape: Vec<usize>, device: Device, requires_grad: bool) -> Tensor<T> {
        let mut rng = rand::rng();
        let data = (0..shape.iter().product()).map(|_| T::from(rng.random::<f32>()).unwrap()).collect(); 
        Tensor::new(db, data, shape, device, requires_grad)
    }

   fn match_ops(&self, d: &Tensor<T>, dx: &Tensor<T>, inputs: &Vec<&Tensor<T>>, device: Device) -> Tensor<T> { 
        match d.op {
            Ops::AddEnum => {
                AddOp::backward(inputs, Some(dx), device)
            },
            Ops::SubEnum => {
                SubOp::backward(inputs, Some(dx), device)
            },
            Ops::MulEnum => {
                MulOp::backward(inputs, Some(dx), device)
            },
            Ops::SinEnum => {
                SinOp::backward(inputs, Some(dx), device)
            },
            Ops::CosEnum => {
                CosOp::backward(inputs, Some(dx), device)
            },
            Ops::PowEnum => {
                PowOp::backward(inputs, Some(dx), device)
            },
            Ops::LnEnum => {
                LnOp::backward(inputs, Some(dx), device)
            },
            Ops::DivEnum => {
                DivOp::backward(inputs, Some(dx), device)
            },
            Ops::MatMulEnum => {
                MatMulOp::backward(inputs, Some(dx), device)
            },
            _ => panic!("Operation not implemented")
        }
    }

    pub fn backward(&self, dx: Option<Vec<i32>>, device: Device) -> HashMap<i32, Tensor<T>> {
        if device == Device::CPU {
            let mut db_mut = self.dtype.write().unwrap();
            for node in db_mut.get_all_mut() {
                node.cpu();
            }
            drop(db_mut);
        }


        let mut all_leaves = Vec::new();
        {
            let db = self.dtype.read().unwrap();
            match dx {
                Some(x) => {
                    // println!("dx: {:?}", x);
                    for node in x {
                        all_leaves.push(node);
                    }
                },
                None => {
                    for node in self.op_chain.nodes() {
                        let outgoing_edges = self.op_chain.edges_directed(node, Outgoing);
                        if outgoing_edges.count() == 0 && db.get(node).unwrap().requires_grad {
                            all_leaves.push(node);
                        }
                    }
                }
            }
        }
        let mut paths = HashMap::new(); 
        for leaf in all_leaves.clone() {
            let path = algo::all_simple_paths::<Vec<_>, _, std::hash::RandomState>(&self.op_chain, self.id, leaf, 0, None).collect::<Vec<_>>();
            paths.insert(leaf, path);
        }
        // println!("All paths: {:?}", paths);

        let mut grad = HashMap::new();

        for leaf in all_leaves.clone() {
            let data = vec![T::from(1).unwrap(); self.data.len()];
            let mut new_graph = GraphMap::<i32, i32, Directed>::new();
            new_graph.add_node(self.id);
            let mut new_tensor = Tensor {
                id: self.id,
                data,
                shape: self.shape.clone(),
                device: Device::CPU,
                op: self.op.clone(),
                requires_grad: self.requires_grad,
                op_chain: new_graph,
                op_head: self.id,
                dtype: self.dtype.clone(),
                cuda_ptr: None
            };
            if device == Device::CUDA {
                new_tensor.cuda();
            }
            grad.insert(leaf, new_tensor);
        }

        for leaf in all_leaves.clone() {
            // println!("Leaf: {}", grad[&leaf]);
            let path = &paths[&leaf];
            let mut arr: Vec<Tensor<T>> = Vec::new();
            for p in path {
                let mut temp = grad[&leaf].clone();
                grad.get_mut(&leaf).unwrap().op_head = grad.get(&leaf).unwrap().id;
                for i in 0..p.len() - 1 {
                    // println!("d{:?}/d{:?}", p[i], p[i + 1]);
                    let neighbor = self.op_chain.neighbors_directed(p[i], Outgoing).collect::<Vec<_>>();
                    // println!("{:?} = {:?} + {:?}", p[i], neighbor[0], neighbor[1]);
                    let mut inputs = vec![];
                    {
                        let db = self.dtype.read().unwrap();
                        for n in neighbor {
                            // println!("n: {:?}", n);
                            inputs.push(db.get(n).unwrap());
                            // println!("inputs: ");
                        }
                        // let inputs = vec![db.get(neighbor[0]).unwrap(), db.get(neighbor[1]).unwrap()];
                        let op_type = db.get(p[i]).unwrap().op.clone();
                        let input_shapes = inputs.iter().map(|x| x.shape.len()).collect::<Vec<_>>();
                        // println!("derivative of {} w.r.t {}",db.get(p[i]).unwrap(), db.get(p[i+1]).unwrap());
                        let d = db.get(p[i]).unwrap();
                        let dx = db.get(p[i + 1]).unwrap();
                        // println!("d: {}", d);
                        // println!("dx: {}", dx);
                        let d_shape = d.shape.clone();
                        let dx_shape = dx.shape.clone();
                        let output = self.match_ops(d, dx, &inputs, device);
                        // output.cpu();
                        // println!("output: {}", output);
                        let grad_index = inputs.iter().position(|&x| x.id == db.get(p[i+1]).unwrap().id).unwrap();
                        drop(db);
                        if let Ops::MatMulEnum = op_type {
                            // output is b_t and temp is downstream so follow upstream dot b_t
                            // println!("{} {:?}", temp, temp.device);
                            // println!("output: {} {:?}", output, output.device);
                            if input_shapes[0] > 1 || input_shapes[1] > 1 {
                                if grad_index == 0 {
                                    temp = MatMulOp::forward(&vec![&temp, &output]);
                                } else {
                                    temp = MatMulOp::forward(&vec![&output, &temp]);
                                }
                                // println!("temp: {}", temp);
                                if device == Device::CPU {
                                    let mut reduce_ctn = 0;
                                    // println!("d_shape: {:?}, dx_shape: {:?}", d_shape, dx_shape);
                                    if d_shape.len() > 2 && dx_shape.len() > 2 {
                                        for (a, b) in d_shape[0..d_shape.len() - 2].iter().zip(dx_shape[0..dx_shape.len() - 2].iter()) {
                                            reduce_ctn += (a != b) as i32;
                                        }
                                    }
                                    let mut got = ArrayD::from_shape_vec(IxDyn(&temp.shape), temp.data.clone()).unwrap();
                                    let mut corrected_shape = got.shape().to_vec();
                                    for i in 0..reduce_ctn {
                                        // println!("reducing axis: {}", i);
                                        got = got.sum_axis(ndarray::Axis(0));
                                        corrected_shape[i as usize] = 1;
                                    }
                                    temp.data = got.iter().map(|x| *x).collect();
                                    temp.shape = corrected_shape;
                                } else {
                                    let mut reduce_ctn = 0;
                                    // println!("d_shape: {:?}, dx_shape: {:?}", d_shape, dx_shape);
                                    let mut og_shape = temp.shape.clone();
                                    let mut index = 0;
                                    if d_shape.len() > 2 && dx_shape.len() > 2 {
                                        for (a, b) in d_shape[0..d_shape.len() - 2].iter().zip(dx_shape[0..dx_shape.len() - 2].iter()) {
                                            if a != b {
                                                reduce_ctn += max(*a, *b) as i32;
                                                og_shape[index] = 1;
                                            }
                                            index += 1;
                                        }
                                    }
                                    if reduce_ctn > 0 {
                                        unsafe {
                                            let mut data: f32 = 0.0;
                                            let mut ptr_to_data: *mut f32 = &mut data;
                                            println!("reduce_ctn: {}", reduce_ctn);
                                            println!("temp.shape: {:?}", temp.shape);
                                            reduce(reduce_ctn, temp.shape[temp.shape.len()-2].try_into().unwrap(), temp.shape[temp.shape.len()-1].try_into().unwrap(), temp.cuda_ptr.unwrap(), &mut ptr_to_data);
                                            temp.cuda_ptr = Some(ptr_to_data);
                                        }
                                        temp.shape = og_shape;
                                        temp.data = vec![T::from(0.0).unwrap(); temp.shape.iter().product()];
                                    }
                                }
                            } else {
                                temp = MulOp::forward(&vec![&output, &temp]);
                            }
                        } else {
                            temp = MulOp::forward(&vec![&output, &temp]);
                            // println!("temp: {}", temp);
                        }
                    }
                    // println!("output: ");
                    // println!("grad: ");
                    grad.get_mut(&leaf).unwrap().op_chain.add_edge(p[i], p[i + 1], 0);
                    // println!("grad: ");
                }
                arr.push(temp);
            }
            let mut sum = arr[0].clone();
            for i in 1..arr.len() {
                sum = AddOp::forward(&vec![&sum, &arr[i].clone()]);
            }
            // grad.get_mut(&leaf).unwrap().data = sum.data;
            // grad.get_mut(&leaf).unwrap().shape = sum.shape;
            *grad.get_mut(&leaf).unwrap() = sum.clone();
        }


        grad
    }

    pub fn clear_graph(&mut self) {
        self.op_chain = GraphMap::new();
        self.op_chain.add_node(self.id);
        self.op_head = self.id;
    }

    pub fn t(&self) -> Tensor<T> { // TODO: Should I change OpChain?
        let b_arr = ArrayD::from_shape_vec(IxDyn(&self.shape), self.data.clone()).unwrap();
        let b_shape = b_arr.shape();
        let b_t: ArrayD<T>;
        let b_t_shape: Vec<usize>;

        if b_shape.len() == 1 {
            b_t = b_arr; 
            b_t_shape = b_t.shape().to_vec();
        }
        else if b_shape.len() == 2 {
            b_t = b_arr.t().to_owned();
            b_t_shape = b_t.shape().to_vec();
        } else if b_shape.len() == 3 {
            b_t = b_arr.permuted_axes(IxDyn(&[0, 2, 1])).to_owned();
            b_t_shape = b_t.shape().to_vec();
        } else if b_shape.len() == 4 {
            todo!();
        } else {
            panic!("Matrix multiplication grad only supported for <=3D tensors");
        }
        return Tensor {
            id: make_node_uid(),
            data: b_t.iter().map(|x| *x).collect(),
            shape: b_t_shape,
            ..self.clone()
        }
    }
}

impl<T: std::fmt::Debug> std::fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        fn print_recursive<T: std::fmt::Debug>(f: &mut std::fmt::Formatter, shape: &[usize], array: &[T], idx: &mut usize, depth: usize) -> std::fmt::Result {
            if depth == shape.len() {
                write!(f, "{:?} ", array[*idx])?;
                *idx += 1;
            } else {
                let size = shape[depth];
                write!(f, "[")?;
                for i in 0..size {
                    print_recursive(f, shape, array, idx, depth + 1)?;
                    if i < size - 1 {
                        write!(f, ", ")?;
                    }
                }
                write!(f, "]")?;
            }
            Ok(())
        }

        write!(f, "Tensor<{}>(", print_type_of(&self.data[0]))?;
        let mut idx = 0;
        print_recursive(f, &self.shape, &self.data, &mut idx, 0)?;
        write!(f, ", shape=")?;
        write!(f, "{:?})", self.shape)?;
        write!(f, "\n")?; 
        Ok(())
    }
}
