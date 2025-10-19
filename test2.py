import torch

A = torch.tensor([
    [1, 2, 3],
    [4, 5, 6],
], device='cuda:0')  # shape [2, 3, 2] (B, H, W)
print(A.shape)
print(A.stride())
print(A.view(-1))  # A.flatten()
# tensor([1., 2., 3., 4., 5., 6., 1., 2., 3., 4., 5., 6.], device='cuda:0')

B = A.t().contiguous()  # [B, W, H]
print(B.shape)
print(B.stride())
print(B.view(-1))
# tensor([1., 4., 2., 5., 3., 6., 1., 4., 2., 5., 3., 6.], device='cuda:0')

