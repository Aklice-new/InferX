import numpy as np
import torch


def my_broadcast(A:np.ndarray, B:np.ndarray):

    A = A.reshape(-1)  # [3, 1, 4]
    B = B.reshape(-1)  # [1, 5, 1]

    A_strides = [4, 0, 1] 
    B_strides = [0, 1, 0]  

    new_out = np.zeros(3 * 5 * 4).reshape(-1)

    new_strides = [20, 4, 1] # [3, 5, 4]

    for i in range(0, 3):
        for j in range(0, 5):
            for k in range(0, 4):
                # print(i, j, k)
                a = A[i * A_strides[0] + j * A_strides[1] + k * A_strides[2]]
                b = B[i * B_strides[0] + j * B_strides[1] + k * B_strides[2]]
                new_out[i * new_strides[0] + j * new_strides[1] + k * new_strides[2]] = a + b

    return new_out.reshape(3, 5, 4)

if __name__ == '__main__':
    A = np.arange(12).reshape(3, 1, 4)
    B = np.arange(5).reshape(5, 1)

    res = my_broadcast(A, B)

    if np.allclose(res, A + B) == False:
        print("Error")

    print(res, A+B)