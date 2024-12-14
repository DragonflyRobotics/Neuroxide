# import torch
# import numpy as np
#
# a = np.array([1, 2, 3, 4, 5, 6])
# b = np.array([5, 6, 7, 8, 9, 10])
# a = a.reshape(1, 2, 3)
# b = b.reshape(1, 3, 2)
#
# a = torch.tensor(a, dtype=torch.float32, requires_grad=True)
# b = torch.tensor(b, dtype=torch.float32, requires_grad=False)
#
# c = torch.bmm(a, b)
# print("C", c)
# grad_output = torch.ones_like(c)
# grad = torch.autograd.grad(c, a, torch.ones_like(c))
#
# # transpose last two dimensions of b
# b = torch.transpose(b, 1, 2)
# print(f"{grad_output} * {b}")
# grad_manual = torch.bmm(grad_output, b)
# print("Grad", grad)
# print("Grad Manual", grad_manual)
#
import numpy as np

# Define two 1D vectors
a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
b = np.array([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])

# Forward pass: Compute dot product
c = np.dot(a, b)
print(f"Dot product: {c}")

# Backward pass: Gradients
grad_c_wrt_a = b  # Gradient of c with respect to a
grad_c_wrt_b = a  # Gradient of c with respect to b

print(f"Gradient with respect to a: {grad_c_wrt_a}")
print(f"Gradient with respect to b: {grad_c_wrt_b}")

#get grad using autograd
import torch
a = torch.tensor(a, dtype=torch.float32, requires_grad=True)
b = torch.tensor(b, dtype=torch.float32, requires_grad=False)

c = torch.dot(a, b)
c.backward()

print(a.grad)

