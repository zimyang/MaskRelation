"""
Custom Loss Functions for MindSpore platform.
"""
import mindspore
from mindspore import nn, ops
    
def loss_tc(T_attention):
    batch_size, T, A, H, W = T_attention.shape
    loss_tc = 0
    for t in range(T-1):
        mapi = T_attention[:, t, :, :, :]
        mapj = T_attention[:, t+1, :, :, :]
        loss_tc += ops.dist(mapi, mapj,p=1)
    loss_tc = loss_tc / T / batch_size
    return loss_tc
    
def loss_od(T_att_features):
    eps=1e-8
    # [B, T, A, H, W]
    # B, T, A, C = T_att_features.shape
    matrix_A = T_att_features
    a_n = matrix_A.norm(dim=2).unsqueeze(2)
    # a_n [B, N, 1]
    # Normalize    
    a_norm = matrix_A / ops.max(a_n, eps * ops.ones_like(a_n))
    # patch-wise absolute value of cosine similarity
    sim_matrix = ops.einsum('abc,acd->abd', a_norm, a_norm.swapaxes(1,2))
    loss_rc = sim_matrix.mean()
    return loss_rc

class CustomLossCell(nn.Cell):
    def __init__(self, backbone, loss_fn):
        super(CustomLossCell, self).__init__(auto_prefix=False)
        self._backbone = backbone
        self.loss_ce = loss_fn
    
    def construct(self, data, label):
        logits, T_attention, T_att_features = self._backbone(data)
        loss_ce = self.loss_ce(logits, label)
        loss_temporal_cons = loss_tc(T_attention)
        loss_orthogon_dive = loss_od(T_att_features)
        return loss_ce+loss_orthogon_dive+loss_temporal_cons