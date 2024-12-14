import torch

a = torch.randn(2, 3, requires_grad=True)
b = torch.randn(3, 2)
print("A", a)
print("B", b)

c = torch.mm(a, b)
print("C", c)
grad_output = torch.ones_like(c)
grad = torch.autograd.grad(c, a, torch.ones_like(c))
grad_manual = torch.mm(grad_output, b.t())
print(grad)
print(grad_manual)
