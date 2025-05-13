import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

# Set device (CPU to match the Rust code)
device = torch.device("cuda")

# Define the model
# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.linear1 = nn.Linear(16, 16)
#         self.linear2 = nn.Linear(16, 16)
#
#     def forward(self, x):
#         x = self.linear1(x)
#         x = self.linear2(x)
#         return x

# Initialize model, optimizer, loss function
# model = SimpleNet().to(device)
weights1 = torch.ones((16, 16), device=device)
weights2 = torch.ones((16, 16), device=device)
bias1 = torch.zeros((16,), device=device)
bias2 = torch.zeros((16,), device=device)
weights1 = torch.nn.Parameter(weights1)
weights2 = torch.nn.Parameter(weights2)
bias1 = torch.nn.Parameter(bias1)
bias2 = torch.nn.Parameter(bias2)

optimizer = torch.optim.SGD([weights1, weights2, bias1, bias2], lr=0.0000001)
criterion = nn.MSELoss()

# Training loop
start_time = time.time()
for epoch in range(1500):
    num = random.randint(0, 99)
    input_tensor = torch.tensor([[float(num)] * 16], dtype=torch.float32, device=device)
    target_tensor = torch.tensor([[float(num * 2)] * 16], dtype=torch.float32, device=device)

    optimizer.zero_grad()
    # output = torch.nn.functional.linear(input_tensor, weights1, bias1)
    # output = torch.nn.functional.linear(output, weights2, bias2)
    output = torch.matmul(input_tensor, weights1.t()) + bias1
    output = torch.matmul(output, weights2.t()) + bias2
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()

    print(f"Epoch: {epoch} Loss: {loss.item()}")

# Test output
test_input = torch.tensor([[4.0] * 16], dtype=torch.float32, device=device)
with torch.no_grad():
    test_output = torch.nn.functional.linear(test_input, weights1, bias1)
    test_output = torch.nn.functional.linear(test_output, weights2, bias2)
print(test_output)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.4f} seconds")

