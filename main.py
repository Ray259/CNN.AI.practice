import numpy as np


def relu(x):
    return np.maximum(x, 0)


def max_pooling(input_data, pool_size):
    input_height, input_width = input_data.shape
    pool_height, pool_width = pool_size
    output_height = input_height // pool_height
    output_width = input_width // pool_width
    output = np.zeros((output_height, output_width))
    for i in range(output_height):
        for j in range(output_width):
            output[i, j] = np.max(
                input_data[i * pool_height:(i + 1) * pool_height, j * pool_width:(j + 1) * pool_width])
    return output


def avg_pool(input, pool_size, stride):
    in_h, in_w = input.shape
    out_size = (in_h - pool_size) // stride + 1
    out = np.zeros((out_size, out_size))
    t = 0
    for i in range(out_size):
        for j in range(out_size):
            for k in range(i, 1, pool_size):
                for l in range(j, 1, pool_size):
                    t += input[k, l]
            out[i, j] = t / (pool_size * pool_size)
    t = 0
    return out


def convol(input, kernel, stride, padding):
    in_h, in_w = input.shape
    k_h, k_w = kernel.shape
    out_size = (in_h - k_h + 2 * padding) // stride + 1
    out = np.zeros((out_size, out_size))
    for i in range(out_size):
        for j in range(out_size):
            out[i, j] = np.sum(input[i:i + k_h, j:j + k_w] * kernel)
    return out


input_data = np.array([[1, 2, 0, 1],
                       [0, 1, 2, 3],
                       [3, 0, 1, 2],
                       [2, 1, 0, 1]])
kernel = np.array([[1, 0],
                   [0, 1]])

conv_output = convol(input_data, kernel, 1, 0)
conv_output_relu = relu(conv_output)
pool_output = avg_pool(conv_output_relu, 2, 1)

print(conv_output)
print(conv_output_relu)
print(pool_output)
