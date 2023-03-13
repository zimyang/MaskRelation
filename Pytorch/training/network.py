import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorchcv.model_provider import get_model
from timm.models.layers import trunc_normal_

from training.r3d import mc3_18
from utils.graph_conv import *
from utils.util import pretrained_model
from einops import rearrange

def define_network(f_dim, h_didm, a_dim, config):
    model = Model(f_dim, h_didm, a_dim, config)
    return model

class MC3(nn.Module):
    def __init__(self):
        super(MC3, self).__init__()
        base = models.video.mc3_18(pretrained=False)
        r3d_state = torch.load(pretrained_path['mc3_18'])
        base.load_state_dict(r3d_state)

        self.base = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(512, 2)
    
    def forward(self, clip):
        x = self.base(clip).flatten(1)
        return self.fc(x)


class Model(nn.Module):
    def __init__(self, f_dim, h_dim, a_dim, config):
        super(Model, self).__init__()
        self.f_dim = f_dim 
        self.h_dim = h_dim 
        self.a_dim = a_dim 
        self.mask_strategy = config.mask_strategy
        self.masked_drop = config.mask_rate

        self.edges = nn.Parameter(random_edges(a_dim))
        
        self.enc = Encoder()

        self.atn = AttentionMap(in_channels=f_dim, out_channels=a_dim)
        self.drop = nn.Dropout1d(config.dropout_rate)
        self.atp = AttentionPooling() # MaxAttentionPooling() 
        self.drop2d = nn.Dropout(config.dropout_rate)

        self.cell = TGCN(f_dim, h_dim)
        
        self.bn = nn.BatchNorm1d(a_dim * h_dim) 
        self.lrelu = nn.ReLU(True)
        self.fc = nn.Linear(a_dim * h_dim, 2)
        
        self.init_weights()
    
    def init_weights(self):
        trunc_normal_(self.fc.weight, std=.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, clip):
        adj = self.edges
        if self.training and self.mask_strategy!='none':
            adj = MaskConnection(adj, self.mask_strategy, self.masked_drop)
        adj = calculate_laplacian_with_self_loop(adj)

        batch_size = clip.shape[0]
        h1 = torch.zeros(batch_size, self.a_dim, self.h_dim).type_as(clip)
        h2 = torch.zeros(batch_size, self.a_dim, self.h_dim).type_as(clip)
        
        out = None
        snippets = torch.chunk(clip, chunks=5, dim=2)
        for index, input in enumerate(snippets):
            ## Features
            feature_map = self.enc(input)
            
            attention_maps_ = self.atn(feature_map)
        
            dropout_mask = self.drop(torch.ones([attention_maps_.shape[0], self.a_dim, 1], device=input.device))
            
            attention_maps = attention_maps_ * torch.unsqueeze(dropout_mask, -1) 
            feature_matrix_ = self.atp(feature_map, attention_maps) 
            feature_matrix = feature_matrix_ * dropout_mask 

            h1, h2 = self.cell(feature_matrix, h1, h2, adj)

            if index > 0:
                T_attention_maps = torch.cat((T_attention_maps, attention_maps_.unsqueeze(1)), dim=1)
                T_att_features = torch.cat((T_att_features, attention_maps_.unsqueeze(1).flatten(-2)), dim=1)
                
            else: 
                T_attention_maps = attention_maps_.unsqueeze(1)
                T_att_features = attention_maps_.unsqueeze(1).flatten(-2)
                
        out = h2
        T_att_features = T_att_features.flatten(0,1)
        
        x = out.flatten(1)
        
        x = self.bn(x)
        x = self.lrelu(x)
        
        logits = self.fc(x)

        if self.training:
            return logits, self.loss_tc(T_attention_maps), self.loss_od(T_att_features)
        return logits
    
    def loss_tc(self, T_attention):
        batch_size, T, A, H, W = T_attention.size()
        loss_tc = 0
        for t in range(T-1):
            mapi = T_attention[:, t, :, :, :]
            mapj = T_attention[:, t+1, :, :, :]
            loss_tc += torch.dist(mapi, mapj,p=1)
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
        a_norm = matrix_A / torch.max(a_n, eps * torch.ones_like(a_n))
        # patch-wise absolute value of cosine similarity
        sim_matrix = torch.einsum('abc,acd->abd', a_norm, a_norm.transpose(1,2))
        loss_rc = sim_matrix.mean()
        return loss_rc

    def l2_reg(self):
        reg_loss = 0.0
        for param in self.cell.parameters():
            reg_loss += torch.sum(param ** 2) / 2
        reg_loss = 1.5e-3 * reg_loss
        return reg_loss
    
    def attention(self):
        return {'img': self.img[:, :, :, :, :], 'attention': F.interpolate(self.T_attention[:, :, :, :, :], (self.T_attention.shape[-3], self.img.shape[-2], self.img.shape[-1]))} 
    

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        
        base = mc3_18(pretrained=False)
        pretrain_path = pretrained_path['mc3_18']

        pretrained_model(base, pretrain_path)

        self.base = base

    def forward(self, clip):
        return self.base(clip)

class AttentionMap(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentionMap, self).__init__()
        
        self.num_attentions = out_channels
        
        self.conv_extract = nn.Conv3d(in_channels, in_channels, kernel_size=(7,3,3), stride=(3, 1, 1), padding=1) 
        self.bn1 = nn.BatchNorm3d(in_channels)
        self.conv2 = nn.Conv3d(in_channels, out_channels, kernel_size=(3, 1, 1), stride=(2, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels) 

    def forward(self, x):
        if self.num_attentions==0:
            return torch.ones([x.shape[0],1,1,1],device=x.device)
        x = self.conv_extract(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = self.bn2(x)

        x = F.adaptive_avg_pool3d(x, (1, None, None)) 
        x = x.squeeze(2)
        x = F.gelu(x)
        return x

class AttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, features, attentions,norm=2):
        H, W = features.size()[-2:]
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions=F.interpolate(attentions,size=(H,W), mode='bilinear', align_corners=True)
        if norm==1:
            attentions=attentions+1e-8
        if len(features.shape)==4:
            feature_matrix=torch.einsum('imjk,injk->imn', attentions, features)
        else:
            feature_matrix=torch.einsum('imjk,indjk->imn', attentions, features)
        if norm==1:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)
            feature_matrix/=w
        if norm==2:
            feature_matrix = F.normalize(feature_matrix,p=2,dim=-1)
        if norm==3:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)+1e-8
            feature_matrix/=w
        return feature_matrix

class MaxAttentionPooling(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, features, attentions,norm=2):
        H, W = features.size()[-2:]
        
        B, M, AH, AW = attentions.size()
        if AH != H or AW != W:
            attentions=F.interpolate(attentions,size=(H,W), mode='bilinear', align_corners=True)
        if norm==1:
            attentions=attentions+1e-8
        if len(features.shape)==4:
            feature_matrix=torch.einsum('imjk,injk->imnjk', attentions, features)
        else:
            feature_matrix=torch.einsum('imjk,indjk->imnjk', attentions, features)
        if norm==1:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)
            feature_matrix/=w
        if norm==2:
            feature_matrix = F.max_pool3d(feature_matrix, [1,2,2], [1,2,2])
            feature_matrix = F.max_pool3d(feature_matrix, [1,2,2], [1,2,2])
            feature_matrix = feature_matrix.squeeze()
            feature_matrix = F.normalize(feature_matrix, p=2, dim=-1)
        if norm==3:
            w=torch.sum(attentions,dim=(2,3)).unsqueeze(-1)+1e-8
            feature_matrix/=w
        return feature_matrix

class TGCN(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(TGCN, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        self.tgcn_layer1 = TGCNCell(self.in_dim, self.h_dim)
        self.tgcn_layer2 = TGCNCell(self.h_dim, self.h_dim)

    def forward(self, inputs, H1, H2, adj):
        batch_size, a_dim, c = inputs.shape
        assert self.in_dim == c
        H1 = self.tgcn_layer1(inputs, H1, adj)
        H2 = self.tgcn_layer2(H1, H2, adj)
        return H1, H2

class TGCNCell(nn.Module):
    def __init__(self, in_dim, h_dim):
        super(TGCNCell, self).__init__()
        self.in_dim = in_dim
        self.h_dim = h_dim
        
        self.gconv1 = GConv(in_dim + h_dim, h_dim * 2, bias=True)
        self.gconv2 = GConv(in_dim + h_dim, h_dim, bias=False)

    def forward(self, inputs, hidden_state, adj):
        concatenation = torch.sigmoid(self.gconv1(inputs, hidden_state, adj))
        
        r, u = torch.chunk(concatenation, chunks=2, dim=2)
        
        c = torch.tanh(self.gconv2(inputs, r * hidden_state, adj))
        
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        
        return new_hidden_state

class GConv(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """
    def __init__(self, in_features, out_features, bias=True):
        super(GConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        nn.init.xavier_normal_(self.weight)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, inputs, hidden_state, laplacian):
        # [x, h] (batch_size, a_dim, input_dim + h_dim)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        batch_size, a_dim, ih_dim = concatenation.shape
        # [x, h] (a_dim, batch_size * (input_dim + h_dim))
        concatenation = rearrange(concatenation, 'b a c -> a (b c)')
        # A[x, h] (a_dim, batch_size * (input_dim + h_dim))
        a_times_concat = laplacian @ concatenation
        # A[x, h] (batch_size, a_dim, input_dim + h_dim)
        a_times_concat = rearrange(a_times_concat, 'a (b c) -> b a c', b=batch_size, c=ih_dim)
        output = a_times_concat @ self.weight
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        base = get_model('Xception', pretrained=False)
        pretrain_state = torch.load(pretrained_path['xception'])
        base.load_state_dict(pretrain_state)
        
        base = nn.Sequential(*list(base.children())[:-1])
        base[0].final_block.pool = nn.Sequential(nn.AdaptiveAvgPool2d((1,1)))
        self.base = base
        self.flatten = nn.Flatten(1)
        self.bn = nn.BatchNorm1d(2048)
        self.fc = nn.Linear(2048, 2)

    def forward(self, images):
        out = torch.zeros((images.shape[0], 2048), device=images.device)
        for img in torch.split(images, 1, 2):
            img = img.squeeze(2)
            f = self.base(img)
            f = self.flatten(f)
            out += f
        out /= images.shape[2]
        out = self.bn(out)
        return self.fc(out)


pretrained_path = {
    'resnet18': 'pretrained/resnet18.pth',
    'resnet50': 'pretrained/resnet50.pth',
    'xception': 'pretrained/xception.pth',
    'arcface': 'pretrained/model_ir_se50.pth',
    'r3d_18': 'pretrained/r3d_18.pth',
    'r2plus1d_18': 'pretrained/r2plus1d_18.pth',
    'mc3_18': 'pretrained/mc3_18.pth'
}

def MaskConnection(adj, mask_strategy, masked_rate):
    if masked_rate > 0.:
        if mask_strategy == 'min':
            min_ = adj.min()
            max_ = adj.max()
            q = min_ + (max_ - min_) * masked_rate
            drop_matrix = torch.gt(adj, q)
        elif mask_strategy == 'random':
            q_matrix = F.dropout(adj, masked_rate)
            drop_matrix = torch.gt(q_matrix, 0)
        else:
            print(f"No such strategy {mask_strategy}.")
        adj = drop_matrix * adj
    return adj
