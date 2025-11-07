from kernels import vector_add_stream, matrix_transpose
import numpy as np

if __name__ == "__main__":
    # create 4x4 matrix of elements 1 - 16
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32)
    print(a)
    print(a.shape)

    out = matrix_transpose(a)
    print(out)
