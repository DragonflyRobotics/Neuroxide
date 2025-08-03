import torch

# Flat data like in your Rust vectors
a_data = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    7.0, 8.0, 9.0, 10.0, 11.0, 12.0
]
b_data = [
    1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
    7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0,
    19.0, 20.0, 21.0, 22.0, 23.0, 24.0,
    25.0, 26.0, 27.0, 28.0, 29.0, 30.0,
    31.0, 32.0, 33.0, 34.0, 35.0, 36.0
]

# Create flat tensors
a = torch.tensor(a_data, dtype=torch.float32, requires_grad=True)
b = torch.tensor(b_data, dtype=torch.float32)

# Reshape according to your Rust tensor shapes
# a shape: [2, 1, 2, 3]
a = a.view(2, 1, 2, 3)

# b shape: [2, 3, 2, 2]
b = b.view(2, 3, 3, 2)

a = torch.tensor(a.detach().numpy(), dtype=torch.float32, requires_grad=True)
b = torch.tensor(b.detach().numpy(), dtype=torch.float32)

result = torch.matmul(a, b)
# get grad of a w.r.t result 
result.backward(torch.ones_like(result))
print(a.grad)  # should print the gradient of a

print("Result shape:", result.shape)  # should be (2,3,3,2)
print(result)

#
# import torch
#
# # Simulate flat data arrays (as if from Rust)
# a_data = [
#     1.0, 2.0, 3.0,
#     4.0, 5.0, 6.0,
#     1.0, 2.0, 3.0,
#     4.0, 5.0, 6.0,
# ]  # Total: 12 values
# b_data = [
#     5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
# ]  # Total: 36 values
#
# # Create tensors and reshape like in Rust
# a = torch.tensor(a_data, dtype=torch.float32, requires_grad=True).reshape(2, 2, 3)
# b = torch.tensor(b_data, dtype=torch.float32).reshape(1, 3, 2)
#
# a = torch.tensor(a.detach().numpy(), dtype=torch.float32, requires_grad=True)
# b = torch.tensor(b.detach().numpy(), dtype=torch.float32)
#
# # Perform batched matrix multiplication
# result = torch.matmul(a, b)  # (2, 3, 2, 2)
#
# # Backward with ones
# result.backward(torch.ones_like(result))
#
# # Assertions
# assert a.grad is not None, "Gradient on 'a' is None"
#
# # Optionally: inspect gradient values
# print("✅ Result shape:", result.shape)
# print("✅ Gradient of a:\n", a.grad)
# print("✅ MatMul result:\n", result)
#
