pub fn broadcast_shapes_linear(shape1: &mut Vec<usize>, shape2: &mut Vec<usize>) {
    // fill with zeros from left to right 
    let diff = shape1.len() as i32 - shape2.len() as i32;
    if diff > 0 {
        for _ in 0..diff {
            shape2.insert(0, 1);
        }
    } else {
        for _ in 0..-diff {
            shape1.insert(0, 1);
        }
    }

    for index in 0..shape1.len() {
        if shape1[index] != shape2[index] {
            if shape1[index] == 1 {
                shape1[index] = shape2[index];
            } else if shape2[index] == 1 {
                shape2[index] = shape1[index];
            } else {
                panic!("Incompatible shapes");
            }
        }
    }
}

pub fn broadcast_shapes_matmul(shape1: &mut Vec<usize>, shape2: &mut Vec<usize>) -> Vec<usize> {
    println!("Got shapes: {:?} and {:?}", shape1, shape2);
    let min_dim = std::cmp::min(shape1.len(), shape2.len());
    let mut result_shape;

    // check for any 0s in the shape
    if shape1.contains(&0) || shape2.contains(&0) {
        panic!("Invalid shape");
    }
    
    if shape1.len() == shape2.len() { // same shape so no expansion needed
        let dims = shape1.len();
        if dims == 1 {
            if shape1[0] != shape2[0] {
                if shape1[0] == 1 {
                    shape1[0] = shape2[0];
                } else if shape2[0] == 1 {
                    shape2[0] = shape1[0];
                } else {
                    panic!("Incompatible shapes");
                }
            }
        } else if dims == 2 {
            if shape1[1] != shape2[0] {
                panic!("Incompatible shapes");
            }
        } else if dims >= 3 {
            for i in 0..dims - 2 {
                if shape1[i] != shape2[i] {
                    if shape1[i] == 1 {
                        shape1[i] = shape2[i];
                    } else if shape2[i] == 1 {
                        shape2[i] = shape1[i];
                    } else {
                        panic!("Incompatible shapes");
                    }
                }
            }
            if shape1[shape1.len()-1] != shape2[shape2.len()-2] {
                panic!("Incompatible shapes");
            }
        }
        result_shape = vec![0; dims];
        if dims >= 2 {
            for i in 0..dims-2 {
                assert_eq!(shape1[i], shape2[i]);
                result_shape[i] = shape1[i];
            }
            result_shape[dims-2] = shape1[dims-2];
            result_shape[dims-1] = shape2[dims-1];
        } else {
            result_shape[0] = 1;
        }
    } else { // Shape lengths are not the same and expansion needed
        let mut reduce_dims = false;
        if min_dim == 1 {
            if shape1.len() == 1 {
                shape1.insert(0, 1);
            } else {
                shape2.push(1);
                reduce_dims = true;
            }
        }
        let diff = shape1.len() as i32 - shape2.len() as i32;
        if diff > 0 {
            for _ in 0..diff {
                shape2.insert(0, 1);
            }
        } else {
            for _ in 0..-diff {
                shape1.insert(0, 1);
            }
        }
        assert_eq!(shape1.len(), shape2.len());
        for index in 0..shape1.len()-2 {
            if shape1[index] != shape2[index] {
                if shape1[index] == 1 {
                    shape1[index] = shape2[index];
                } else if shape2[index] == 1 {
                    shape2[index] = shape1[index];
                } else {
                    panic!("Incompatible shapes");
                }
            }
        }
        if shape1[shape1.len()-1] != shape2[shape2.len()-2] {
            panic!("Incompatible shapes");
        }
        result_shape = vec![0; shape1.len()];
        for i in 0..shape1.len()-2 {
            assert_eq!(shape1[i], shape2[i]);
            result_shape[i] = shape1[i];
        }
        result_shape[shape1.len()-2] = shape1[shape1.len()-2];
        result_shape[shape1.len()-1] = shape2[shape2.len()-1];
        if reduce_dims {
            result_shape.pop();
        }
    }
        

    println!("{:?}", shape1);
    println!("{:?}", shape2);
    println!("{:?}", result_shape);
    result_shape
}

