"""
        let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let layer_1_weights = Tensor::<f32>::new(&db, vec![1.0; 16], vec![1, 16], Device::CPU, true);
    let layer_1_biases = Tensor::<f32>::new(&db, vec![1.0; 16*16], vec![16, 16], Device::CPU, true);
    let layer_2_weights = Tensor::<f32>::new(&db, vec![1.0; 16], vec![16, 1], Device::CPU, true);
    let layer_2_biases = Tensor::<f32>::new(&db, vec![1.0; 16], vec![16, 1], Device::CPU, true);
    let pow_const = Tensor::<f32>::new(&db, vec![2.0; 16], vec![16, 1], Device::CPU, false);
    let input = Tensor::<f32>::new(&db, vec![1.0; 16], vec![16, 1], Device::CPU, false);
    let output = Tensor::<f32>::new(&db, vec![2.0; 16], vec![16, 1], Device::CPU, false);
    let c = matmul!(input, layer_1_weights) + layer_1_biases;
    let c = matmul!(c, layer_2_weights) + layer_2_biases;

    let loss = pow!(c - output, pow_const);
    let grad = loss.backward(None);
    for i in grad.into_iter() {
        println!("{}", i.1);
    }
"""

# import torch
#
# layer_1_weights = torch.ones(1, 16).requires_grad_(True)
# layer_1_biases = torch.ones(16, 16).requires_grad_(True)
# layer_2_weights = torch.ones(16, 1).requires_grad_(True)
# layer_2_biases = torch.ones(16, 1).requires_grad_(True)
# pow_const = torch.ones(16, 1) * 2
# input = torch.ones(16, 1)
# output = torch.ones(16, 1) * 2
# c = torch.matmul(input, layer_1_weights) + layer_1_biases
# c = torch.matmul(c, layer_2_weights) + layer_2_biases
# loss = (c - output).pow(pow_const)
#
# print(loss)
# grad = torch.autograd.grad(loss, [layer_1_weights, layer_1_biases, layer_2_weights, layer_2_biases], grad_outputs=torch.ones_like(loss))
# for i in grad:
#     print(i)
import torch

a = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float, requires_grad=True)
b = torch.tensor([5, 6, 7, 8, 9, 10], dtype=torch.float, requires_grad=True)
c = torch.matmul(a, b)
c.backward()
print(a.grad)
