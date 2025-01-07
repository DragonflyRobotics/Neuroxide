import torch
import time

"""
    {
    let db = Arc::new(RwLock::new(TensorDB::new(DTypes::F32)));
    let mut layer_1_weights = Tensor::<f32>::new(&db, vec![1.0; 16], vec![1, 16], Device::CUDA, true);
    let mut layer_1_biases = Tensor::<f32>::new(&db, vec![1.0; 16*16], vec![16, 16], Device::CUDA, true);
    let mut layer_2_weights = Tensor::<f32>::new(&db, vec![1.0; 16], vec![16, 1], Device::CUDA, true);
    let mut layer_2_biases = Tensor::<f32>::new(&db, vec![1.0; 16], vec![16, 1], Device::CUDA, true);
    let pow_const = Tensor::<f32>::new(&db, vec![2.0; 16], vec![16, 1], Device::CUDA, false);
    let lr = Tensor::<f32>::new(&db, vec![0.0000001], vec![1], Device::CUDA, false);
    for iteration in 0..600 {
        let num: f32 = rand::thread_rng().gen_range(0..100) as f32; 
        let input = Tensor::<f32>::new(&db, vec![num; 16], vec![16, 1], Device::CUDA, false);
        let output = Tensor::<f32>::new(&db, vec![num * 2.0; 16], vec![16, 1], Device::CUDA, false);
        
        let c = add!(matmul!(input, layer_1_weights), layer_1_biases);
        let c = add!(matmul!(c, layer_2_weights), layer_2_biases);
        let loss = pow!(c - output, pow_const);
        
        let grad = loss.backward(None);
        
        layer_1_weights = layer_1_weights.clone() - grad.get(&layer_1_weights.id).unwrap().clone() * lr.clone();
        layer_1_weights.clear_graph();
        
        layer_1_biases = layer_1_biases.clone() - grad.get(&layer_1_biases.id).unwrap().clone() * lr.clone();
        layer_1_biases.clear_graph();
        
        layer_2_weights = layer_2_weights.clone() - grad.get(&layer_2_weights.id).unwrap().clone() * lr.clone();
        layer_2_weights.clear_graph();
        
        layer_2_biases = layer_2_biases.clone() - grad.get(&layer_2_biases.id).unwrap().clone() * lr.clone();
        layer_2_biases.clear_graph();
        println!("Epoch: {} Loss: {}", iteration, loss);
   } 
   
   let input = Tensor::<f32>::new(&db, vec![4.0; 16], vec![16, 1], Device::CUDA, false);
   
   let c = add!(matmul!(input, layer_1_weights), layer_1_biases);
   let c = add!(matmul!(c, layer_2_weights), layer_2_biases);
   println!("{}", c);
}
"""
start = time.time()
device = torch.device("cpu")

with torch.device(device):
# Define the model
    layer_1_weights = torch.ones(1, 16, requires_grad=True)
    layer_1_biases = torch.ones(16, 16, requires_grad=True)
    layer_2_weights = torch.ones(16, 1, requires_grad=True)
    layer_2_biases = torch.ones(16, 1, requires_grad=True)
    lr = torch.tensor([0.0000001])

# Training loop
    for iteration in range(600):
        num = torch.randint(0, 100, (1,), dtype=torch.float32)
        input = torch.ones(16, 1) * num
        output = torch.ones(16, 1) * (num * 2.0)
        
        c = input @ layer_1_weights + layer_1_biases
        c = c @ layer_2_weights + layer_2_biases
        loss = torch.pow(c - output, 2)
        
        grad = torch.autograd.grad(loss, [layer_1_weights, layer_1_biases, layer_2_weights, layer_2_biases], grad_outputs=torch.ones_like(loss))
        
        layer_1_weights = layer_1_weights - grad[0] * lr
        layer_1_biases = layer_1_biases - grad[1] * lr
        layer_2_weights = layer_2_weights - grad[2] * lr
        layer_2_biases = layer_2_biases - grad[3] * lr

            
        # print(f"Epoch: {iteration} Loss: {loss}")

# Inference
    input = torch.ones(16, 1) * 4.0
    c = input @ layer_1_weights + layer_1_biases
    c = c @ layer_2_weights + layer_2_biases
    # print(c)



print(f"Time taken: {time.time() - start}")
