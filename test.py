import torch

a = torch.tensor([1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0], requires_grad=True)
b = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], requires_grad=True)
a = a.reshape(2, 2, 2)
b = b.reshape(2, 2, 3)
c = torch.matmul(a, b)
d = (6*c + 5)**2

grad = torch.autograd.grad(d, a, torch.ones_like(c))
print(grad)



