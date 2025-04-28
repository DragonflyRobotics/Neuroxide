import torch
import time

start = time.time()
for _ in range(20):
    a = torch.randn(4000, 4000).to('cuda')
    b = torch.randn(4000, 4000).to('cuda')
    c = torch.matmul(a, b)
end = time.time()
print(f"Time taken for 2000 matrix multiplications: {end - start:.5f} seconds")


