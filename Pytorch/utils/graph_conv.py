import numpy as np
import torch

def calculate_laplacian_with_self_loop(matrix):    
    matrix = matrix + torch.eye(matrix.size(0), device=matrix.device) 
    row_sum = matrix.sum(1)
    
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0 
    
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = (
        matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    )
    normalized_laplacian = normalized_laplacian.to(matrix.device)
    return normalized_laplacian

def random_edges(dim):
    matrix = np.random.rand(dim, dim)
    matrix = torch.tensor(matrix,dtype=torch.float32)
    matrix = matrix > 0.5
    matrix = matrix.int()
    
    return matrix