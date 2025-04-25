import torch
import torch.nn as nn
import torch.optim as optim
import time
import random

# Set device (CPU to match the Rust code)
device = torch.device("cuda")

# Define the model
class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.linear1 = nn.Linear(16, 16)
        self.linear2 = nn.Linear(16, 16)

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        return x

# Initialize model, optimizer, loss function
model = SimpleNet().to(device)
optimizer = optim.SGD(model.parameters(), lr=1e-7)
criterion = nn.MSELoss()

# Training loop
start_time = time.time()
for epoch in range(1500):
    num = random.randint(0, 99)
    input_tensor = torch.tensor([[float(num)] * 16], dtype=torch.float32, device=device)
    target_tensor = torch.tensor([[float(num * 2)] * 16], dtype=torch.float32, device=device)

    optimizer.zero_grad()
    output = model(input_tensor)
    loss = criterion(output, target_tensor)
    loss.backward()
    optimizer.step()

    # print(f"Epoch: {epoch} Loss: {loss.item()}")

# Test output
test_input = torch.tensor([[4.0] * 16], dtype=torch.float32, device=device)
test_output = model(test_input)
print(test_output)

end_time = time.time()
print(f"Time taken: {end_time - start_time:.4f} seconds")

