import torch

a = torch.tensor([1], dtype=torch.float, requires_grad=True)  # [2,3,2]
b = torch.arange(1, 13, dtype=torch.float32).reshape(2, 3, 2)  # [2,2,3]


a.requires_grad = True
a.retain_grad()
b.requires_grad = True
b.retain_grad()

c = a * b  # [2,3,2]

grad = torch.ones_like(c)
c.backward(gradient=grad)  # ✅ works
print("x.grad:", a.grad)
print("y.grad:", b.grad)

# for _ in range(100000):
#     x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], requires_grad=True)
#     y = torch.tensor([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], requires_grad=True)

#     z1 = x + y
#     z2 = x * y

#     cat0 = torch.cat([z1, z2], dim=0)
#     cat1 = torch.cat([z1, z2], dim=1)

#     slice0 = cat0[1:3, :]
#     slice1 = cat1[:, 2:5]

#     view0 = slice0.reshape(3, 2)
#     view1 = slice1.reshape(3, 2)

#     unsq = view0.unsqueeze(1)
#     sq = unsq.squeeze(1)

#     perm = sq.permute(1, 0)

#     final_tensor = perm + view1.permute(1, 0)
#     # print("final_tensor:", final_tensor)

#     grad = torch.ones_like(y)
#     final_tensor.backward(gradient=grad)  # ✅ works

#     # print("x.grad:", x.grad)
#     # print("y.grad:", y.grad)
