import numpy as np
import mindspore
from mindspore import ops

def calculate_laplacian_with_self_loop(matrix):    
    matrix = matrix + ops.eye(ops.shape(matrix)[0], ops.shape(matrix)[1], mindspore.int32) 
    row_sum = matrix.sum(1)
    
    d_inv_sqrt = ops.pow(row_sum, -0.5).flatten()
    is_inf = ops.IsInf()

    d_inv_sqrt[is_inf(d_inv_sqrt)] = 0.0 
    
    d_mat_inv_sqrt = ops.diag(d_inv_sqrt) # GPU
    # d_inv_sqrt = mindspore.numpy.asarray(d_inv_sqrt)
    # d_mat_inv_sqrt = mindspore.numpy.diag(d_inv_sqrt) # CPU
    
    # normalized_laplacian = (
    #     matrix.matmul(d_mat_inv_sqrt).swapaxes(0, 1).matmul(d_mat_inv_sqrt)
    # )
    normalized_laplacian = ops.matmul(matrix, d_mat_inv_sqrt).swapaxes(0, 1)
    normalized_laplacian = ops.matmul(normalized_laplacian, d_mat_inv_sqrt)
    
    return normalized_laplacian

def random_edges(dim):
    matrix = np.random.rand(dim, dim)
    matrix = mindspore.Tensor(matrix,dtype=mindspore.float32)
    
    # greater = ops.Greater()
    # matrix = matrix > 0.5
    q_matrix = ops.gt(matrix, 0.5)
    zero_like = ops.ZerosLike()
    zeros_matrix = zero_like(matrix)

    matrix = ops.masked_fill(zeros_matrix, q_matrix, 1)

    # matrix = matrix.int()
    
    return matrix