import torch

a = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.float, requires_grad=True)
a = a.view(1, 2, 3)
b = torch.tensor([5, 6, 7], dtype=torch.float, requires_grad=True)
b = b.view(3)
dot = torch.matmul(a, b)
print(dot.shape)
grad = torch.autograd.grad(dot, [b], grad_outputs=torch.ones_like(dot))
print(grad)
