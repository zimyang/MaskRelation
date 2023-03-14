import math
import mindspore
from mindspore import nn, ops
from mindspore.common.initializer import initializer, TruncatedNormal, Zero
from training.r3d import mc3_18
from utils.graph_conv import *
from utils.util import pretrained_model
from einops import rearrange

def define_network(f_dim, h_didm, a_dim, config):
    model = Model(f_dim, h_didm, a_dim, config)
    return model


class Model(nn.Cell):
    def __init__(self, f_dim, h_dim, a_dim, config):
        super(Model, self).__init__()
        self.f_dim = f_dim 
        self.h_dim = h_dim 
        self.a_dim = a_dim 
        self.mask_strategy = config.mask_strategy
        self.masked_drop = config.mask_rate

        self.edges = mindspore.Parameter(random_edges(a_dim))
        
        self.enc = Encoder()

        self.atn = AttentionMap(in_channels=f_dim, out_channels=a_dim)
        self.atp = AttentionPooling() # MaxAttentionPooling() 

        self.cell = TGCN(f_dim, h_dim)
        
        self.flatten = nn.Flatten()
        self.bn = nn.BatchNorm1d(a_dim * h_dim) 
        self.lrelu = nn.ReLU()
        self.fc = nn.Dense(a_dim * h_dim, 2)
        
        self.init_weights()
    
    def init_weights(self):
        self.fc.weight.set_data(initializer(TruncatedNormal(sigma=0.02), self.fc.weight.data.shape))
        self.fc.bias.set_data(initializer(Zero(), self.fc.bias.data.shape))
        

    def construct(self, clip):
        adj = self.edges
        if self.mask_strategy!='none': # self.edges.requires_grad and 
            adj = MaskConnection(adj, self.mask_strategy, self.masked_drop)
        adj = calculate_laplacian_with_self_loop(adj)

        batch_size = clip.shape[0]
        h1 = ops.zeros((batch_size, self.a_dim, self.h_dim), clip.dtype)
        h2 = ops.zeros((batch_size, self.a_dim, self.h_dim), clip.dtype)
        
        out = None
        
        snippets = ops.split(clip, output_num=5, axis=2)
        for index, input in enumerate(snippets):
            ## Features
            feature_map = self.enc(input)
            
            attention_maps = self.atn(feature_map)
            feature_matrix = self.atp(feature_map, attention_maps)

            h1, h2 = self.cell(feature_matrix, h1, h2, adj)

            # if index > 0:
            #     print("index > 0")
            #     T_attention_maps = ops.cat((T_attention_maps, attention_maps_.unsqueeze(1)), axis=1)
            #     T_att_features = ops.cat((T_att_features, attention_maps_.unsqueeze(1).flatten(-2)), axis=1)
                
            # else: 
            #     print("index==0")
            #     T_attention_maps = attention_maps_.unsqueeze(1)
            #     T_att_features = attention_maps_.unsqueeze(1).flatten(-2)
                
        out = h2
        # T_att_features = T_att_features.flatten(0,1)
        
        x = self.flatten(out)
        
        x = self.bn(x)
        x = self.lrelu(x)
        
        logits = self.fc(x)

        return logits#, T_attention_maps, T_att_features
    
    def loss_tc(self, T_attention):
        batch_size, T, A, H, W = T_attention.shape
        loss_tc = 0
        for t in range(T-1):
            mapi = T_attention[:, t, :, :, :]
            mapj = T_attention[:, t+1, :, :, :]
            loss_tc += ops.dist(mapi, mapj,p=1)
        loss_tc = loss_tc / T / batch_size
        return loss_tc
    
    def loss_od(self, T_att_features):
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

    def l2_reg(self):
        reg_loss = 0.0
        for param in self.cell.parameters():
            reg_loss += ops.sum(param ** 2) / 2
        reg_loss = 1.5e-3 * reg_loss
        return reg_loss
    

class Encoder(nn.Cell):
    def __init__(self):
        super(Encoder, self).__init__()
        base = mc3_18(pretrained=False)

        self.base = base

    def construct(self, clip):
        return self.base(clip)

class AttentionMap(nn.Cell):
    def __init__(self, in_channels, out_channels):
        super(AttentionMap, self).__init__()
        
        self.num_attentions = out_channels
        
        self.conv_extract = nn.Conv3d(in_channels, in_channels, kernel_size=(7,3,3), stride=(3, 1, 1), pad_mode="pad", padding=1) 
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1))
        self.bn2 = nn.BatchNorm3d(out_channels) 

    def construct(self, x):
        if self.num_attentions==0:
            return ops.ones((x.shape[0],1,1,1))
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = ops.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)

        x = ops.adaptive_avg_pool3d(x, (1, None, None)) 
        x = x.squeeze(2)
        gelu = ops.GeLU()
        x = gelu(x)
        return x

class AttentionPooling(nn.Cell):
    def __init__(self):
        super().__init__()
    def construct(self, features, attentions,norm=2):
        H, W = features.shape[-2:]
        B, M, AH, AW = attentions.shape
        if AH != H or AW != W:
            attentions = ops.interpolate(attentions,size=(H,W), mode='bilinear', align_corners=True)
        if norm==1:
            attentions=attentions+1e-8
        if len(features.shape)==4:
            einsum1 = ops.Einsum('imjk,injk->imn')
            feature_matrix=einsum1((attentions, features))
        else:
            einsum2 = ops.Einsum('imjk,indjk->imn')
            feature_matrix=einsum2((attentions, features))
        if norm==1:
            w=ops.sum(attentions,dim=(2,3)).unsqueeze(-1)
            feature_matrix/=w
        if norm==2:
            l2_normalize = ops.L2Normalize(axis=-1)
            feature_matrix = l2_normalize(feature_matrix)
        if norm==3:
            w=ops.sum(attentions,dim=(2,3)).unsqueeze(-1)+1e-8
            feature_matrix/=w
        return feature_matrix

class MaxAttentionPooling(nn.Cell):
    def __init__(self):
        super().__init__()
    def construct(self, features, attentions,norm=2):
        H, W = features.shape[-2:]
        
        B, M, AH, AW = attentions.shape
        if AH != H or AW != W:
            attentions=ops.interpolate(attentions,size=(H,W), mode='bilinear', align_corners=True)
        if norm==1:
            attentions=attentions+1e-8
        if len(features.shape)==4:
            einsum1 = ops.Einsum('imjk,injk->imnjk')
            feature_matrix=einsum1((attentions, features))
        else:
            einsum2 = ops.Einsum('imjk,indjk->imnjk')
            feature_matrix=einsum2((attentions, features))
        if norm==1:
            w=ops.sum(attentions,dim=(2,3)).unsqueeze(-1)
            feature_matrix/=w
        if norm==2:
            feature_matrix = ops.max_pool3d(feature_matrix, [1,2,2], [1,2,2])
            feature_matrix = ops.max_pool3d(feature_matrix, [1,2,2], [1,2,2])
            feature_matrix = feature_matrix.squeeze()
            l2_normalize = ops.L2Normalize(axis=-1)
            feature_matrix = l2_normalize(feature_matrix)
        if norm==3:
            w=ops.sum(attentions,dim=(2,3)).unsqueeze(-1)+1e-8
            feature_matrix/=w
        return feature_matrix

class TGCN(nn.Cell):
    def __init__(self, in_dim, h_dim):
        super(TGCN, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.tgcn_layer1 = TGCNCell(self.in_dim, self.h_dim)
        self.tgcn_layer2 = TGCNCell(self.h_dim, self.h_dim)

    def construct(self, inputs, H1, H2, adj):
        batch_size, a_dim, c = inputs.shape
        assert self.in_dim == c
        H1 = self.tgcn_layer1(inputs, H1, adj)
        H2 = self.tgcn_layer2(H1, H2, adj)
        return H1, H2

class TGCNCell(nn.Cell):
    def __init__(self, in_dim, h_dim):
        super(TGCNCell, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        
        self.gconv1 = GConv(in_dim + h_dim, h_dim * 2, bias=True)
        self.gconv2 = GConv(in_dim + h_dim, h_dim, bias=False)

    def construct(self, inputs, hidden_state, adj):
        concatenation = ops.sigmoid(self.gconv1(inputs, hidden_state, adj))
        
        r, u = ops.split(concatenation, output_num=2, axis=2)
        
        c = ops.tanh(self.gconv2(inputs, r * hidden_state, adj))
        
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        
        return new_hidden_state

class GConv(nn.Cell):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = mindspore.Parameter(mindspore.Tensor(np.random.rand(in_features, out_features), mindspore.float32))
        if bias:
            self.bias = mindspore.Parameter(mindspore.Tensor(np.random.rand(out_features), mindspore.float32))
        else:
            self.bias = None
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.set_data(initializer('normal', self.weight.data.shape))
        if self.bias is not None:
            self.bias.set_data(ops.uniform(self.bias.data.shape, mindspore.Tensor(-stdv, mindspore.float32), mindspore.Tensor(stdv, mindspore.float32)))
            
    def construct(self, inputs, hidden_state, laplacian):
        # [x, h] (batch_size, a_dim, input_dim + h_dim)
        concat_op = ops.Concat(axis=2)
        concatenation = concat_op((inputs, hidden_state))
        batch_size, a_dim, ih_dim = concatenation.shape
        # [x, h] (a_dim, batch_size * (input_dim + h_dim))
        concatenation = concatenation.swapaxes(0,1)
        concatenation = concatenation.reshape((a_dim, batch_size*ih_dim))
        # concatenation = rearrange(concatenation, 'b a c -> a (b c)')
        # A[x, h] (a_dim, batch_size * (input_dim + h_dim))
        a_times_concat = ops.matmul(laplacian, concatenation)
        # A[x, h] (batch_size, a_dim, input_dim + h_dim)
        a = a_times_concat.shape[0]
        a_times_concat = a_times_concat.reshape((a, batch_size, ih_dim)).swapaxes(1,0)
        # a_times_concat = rearrange(a_times_concat, 'a (b c) -> b a c', b=batch_size, c=ih_dim)

        output = ops.matmul(a_times_concat, self.weight)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


def MaskConnection(adj, mask_strategy, masked_rate):
    if masked_rate > 0.:
        drop_matrix = None
        if mask_strategy == 'min':
            min_ = adj.min()
            max_ = adj.max()
            q = min_ + (max_ - min_) * masked_rate
            drop_matrix = ops.gt(adj, q)
        elif mask_strategy == 'random':
            q_matrix = ops.dropout(adj, masked_rate)
            drop_matrix = ops.gt(q_matrix, 0)
        else:
            print(f"No such strategy {mask_strategy}.")
        adj = ops.mul(drop_matrix, adj)
    return adj
