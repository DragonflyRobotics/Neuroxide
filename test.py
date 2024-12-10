import numpy as np

# a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])
# b = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24])

# a = a.reshape(2, 2, 6)
# b = b.reshape(2, 6, 2)

a = np.random.rand(1, 5, 10, 24)
b = np.random.rand(1, 5, 24, 10)


c = np.matmul(a, b)
# print(c)


def matmul(a, b):
    initial_shape = a.shape
    final_shape = [a.shape[-2], b.shape[-1]]
    broadcast_shape = [np.prod(a.shape[:-1]), a.shape[-1]]
    print(final_shape, broadcast_shape)
    a = a.reshape(broadcast_shape)
    b = b.reshape(broadcast_shape[::-1])
    print(a.shape, b.shape)
    z = np.matmul(a, b)
    print(z.shape)
    print(np.concatenate([initial_shape[:-2], final_shape], axis=0))
    return z.reshape(np.concatenate([initial_shape[:-2], final_shape], axis=0))


a = np.random.rand(1, 5, 10, 24)
b = np.random.rand(1, 5, 24, 10)
a = a.reshape(10, 1*5*10)
b = b.reshape(1*5*10, 10)
print(a.shape, b.shape)
z = np.matmul(a, b)
print(z.shape)
z.reshape(1, 5, 10, 10)
# print(z)
# z = matmul(a, b)
assert np.array_equal(c, z)
print(np.array_equal(c, z))
