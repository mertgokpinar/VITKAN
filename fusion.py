import torch
import torch.nn as nn
from utils import init_max_weights
import torch.nn.functional as F

class BilinearFusion(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, dim1=32, dim2=32, scale_dim1=1, scale_dim2=1, mmhid=64, dropout_rate=0.25):
        super(BilinearFusion, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2

        dim1_og, dim2_og, dim1, dim2 = dim1, dim2, dim1//scale_dim1, dim2//scale_dim2
        skip_dim = dim1+dim2+2 if skip else 0

        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim2_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim1_og, dim2_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim2_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=dropout_rate)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        init_max_weights(self)

    def forward(self, vec1, vec2):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec2) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec2), dim=1))
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec1, vec2) if self.use_bilinear else self.linear_z2(torch.cat((vec1, vec2), dim=1))
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)

        ### Fusion
        o11 = torch.Tensor(o1.shape[0], 1).fill_(1).to(o1.device)
        o22 = torch.Tensor(o2.shape[0], 1).fill_(1).to(o2.device)
        o1 = torch.cat((o1, o11), 1)
        o2 = torch.cat((o2, o22), 1)
        # o1 = torch.cat((o1, torch.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        # o2 = torch.cat((o2, torch.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1) # BATCH_SIZE X 1024
        out = self.post_fusion_dropout(o12)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2), 1)
        out = self.encoder2(out)
        return out


class TrilinearFusion_A(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=1, dim1=32, dim2=32, dim3=32, scale_dim1=1, scale_dim2=1, scale_dim3=1, mmhid=96, dropout_rate=0.25):
        super(TrilinearFusion_A, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = dim1, dim2, dim3, dim1//scale_dim1, dim2//scale_dim2, dim3//scale_dim3
        skip_dim = dim1+dim2+dim3+3 if skip else 0

        ### Path
        
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim3_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim3_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### Graph
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim2_og, dim3_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim2_og+dim3_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### Omic
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = nn.Bilinear(dim1_og, dim3_og, dim3) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim3_og, dim3))
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1)*(dim3+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        init_max_weights(self)

    def forward(self, vec1, vec2, vec3):
        
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec3) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec3), dim=1)) # Gate Path with Omic
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec2, vec3) if self.use_bilinear else self.linear_z2(torch.cat((vec2, vec3), dim=1)) # Gate Graph with Omic
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = self.linear_z3(vec1, vec3) if self.use_bilinear else self.linear_z3(torch.cat((vec1, vec3), dim=1)) # Gate Omic With Path
            o3 = self.linear_o3(nn.Sigmoid()(z3)*h3)
        else:
            o3 = self.linear_o3(vec3)

        ### Fusion
        o1 = torch.cat((o1, torch.FloatTensor(o1.shape[0], 1).fill_(1)), 1)
        o2 = torch.cat((o2, torch.FloatTensor(o2.shape[0], 1).fill_(1)), 1)
        o1 = o1.unsqueeze(0)

        o2 = o2.unsqueeze(0).transpose(2,1)
        o3 = torch.cat((o3, torch.FloatTensor(o3.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1, o2)
        o123 = torch.bmm(o12.transpose(2,1), o3.unsqueeze(0)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out) 
        return out


class TrilinearFusion_B(nn.Module):
    def __init__(self, skip=1, use_bilinear=1, gate1=1, gate2=1, gate3=1, dim1=32, dim2=32, dim3=32, scale_dim1=1, scale_dim2=1, scale_dim3=1, mmhid=96, dropout_rate=0.25):
        super(TrilinearFusion_B, self).__init__()
        self.skip = skip
        self.use_bilinear = use_bilinear
        self.gate1 = gate1
        self.gate2 = gate2
        self.gate3 = gate3

        dim1_og, dim2_og, dim3_og, dim1, dim2, dim3 = dim1, dim2, dim3, dim1//scale_dim1, dim2//scale_dim2, dim3//scale_dim3
        skip_dim = dim1+dim2+dim3+3 if skip else 0

        ### Path
        self.linear_h1 = nn.Sequential(nn.Linear(dim1_og, dim1), nn.ReLU())
        self.linear_z1 = nn.Bilinear(dim1_og, dim3_og, dim1) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim3_og, dim1))
        self.linear_o1 = nn.Sequential(nn.Linear(dim1, dim1), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### Graph
        self.linear_h2 = nn.Sequential(nn.Linear(dim2_og, dim2), nn.ReLU())
        self.linear_z2 = nn.Bilinear(dim2_og, dim1_og, dim2) if use_bilinear else nn.Sequential(nn.Linear(dim2_og+dim1_og, dim2))
        self.linear_o2 = nn.Sequential(nn.Linear(dim2, dim2), nn.ReLU(), nn.Dropout(p=dropout_rate))

        ### Omic
        self.linear_h3 = nn.Sequential(nn.Linear(dim3_og, dim3), nn.ReLU())
        self.linear_z3 = nn.Bilinear(dim1_og, dim3_og, dim3) if use_bilinear else nn.Sequential(nn.Linear(dim1_og+dim3_og, dim3))
        self.linear_o3 = nn.Sequential(nn.Linear(dim3, dim3), nn.ReLU(), nn.Dropout(p=dropout_rate))

        self.post_fusion_dropout = nn.Dropout(p=0.25)
        self.encoder1 = nn.Sequential(nn.Linear((dim1+1)*(dim2+1)*(dim3+1), mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        self.encoder2 = nn.Sequential(nn.Linear(mmhid+skip_dim, mmhid), nn.ReLU(), nn.Dropout(p=dropout_rate))
        
        init_max_weights(self)

    def forward(self, vec1, vec2, vec3):
        ### Gated Multimodal Units
        if self.gate1:
            h1 = self.linear_h1(vec1)
            z1 = self.linear_z1(vec1, vec3) if self.use_bilinear else self.linear_z1(torch.cat((vec1, vec3), dim=1)) # Gate Path with Omic
            o1 = self.linear_o1(nn.Sigmoid()(z1)*h1)
        else:
            o1 = self.linear_o1(vec1)

        if self.gate2:
            h2 = self.linear_h2(vec2)
            z2 = self.linear_z2(vec2, vec1) if self.use_bilinear else self.linear_z2(torch.cat((vec2, vec1), dim=1)) # Gate Graph with Omic
            o2 = self.linear_o2(nn.Sigmoid()(z2)*h2)
        else:
            o2 = self.linear_o2(vec2)

        if self.gate3:
            h3 = self.linear_h3(vec3)
            z3 = self.linear_z3(vec1, vec3) if self.use_bilinear else self.linear_z3(torch.cat((vec1, vec3), dim=1)) # Gate Omic With Path
            o3 = self.linear_o3(nn.Sigmoid()(z3)*h3)
        else:
            o3 = self.linear_o3(vec3)

        ### Fusion
        # print(o1.get_device())
        # print(torch.Tensor(o1.shape[0], 1).fill_(1).cuda(0).get_device())
        
        o11 = torch.Tensor(o1.shape[0], 1).fill_(1).to(o1.device)
        o22 = torch.Tensor(o2.shape[0], 1).fill_(1).to(o2.device)
        o33 = torch.Tensor(o3.shape[0], 1).fill_(1).to(o3.device)

        o1 = torch.cat((o1, o11), 1)
        o2 = torch.cat((o2, o22), 1)
        o3 = torch.cat((o3, o33), 1)
        #o1 = torch.cat((o1, torch.Tensor(o1.shape[0], 1).fill_(1)), 1)
        #o2 = torch.cat((o2, torch.Tensor(o2.shape[0], 1).fill_(1)), 1)
        #o3 = torch.cat((o3, torch.Tensor(o3.shape[0], 1).fill_(1)), 1)
        o12 = torch.bmm(o1.unsqueeze(2), o2.unsqueeze(1)).flatten(start_dim=1)
        o123 = torch.bmm(o12.unsqueeze(2), o3.unsqueeze(1)).flatten(start_dim=1)
        out = self.post_fusion_dropout(o123)
        out = self.encoder1(out)
        if self.skip: out = torch.cat((out, o1, o2, o3), 1)
        out = self.encoder2(out)
        return out
    
class cross(nn.Module):
    def __init__(self, emb_dim = 32, num_heads = 4):
        super(cross, self).__init__()
        self.dim = 32
        self.head = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first = False)
        
    def forward(self, omic, histo):
        omic_vec = omic.unsqueeze(0)
        histo_vec = histo.unsqueeze(0)
        x, _ = self.head(omic_vec,histo_vec,histo_vec)
        x = x.squeeze(0)
        return x

class selfattention(nn.Module):
    def __init__(self, emb_dim = 64, num_heads = 4):
        super(selfattention, self).__init__()
        
        self.head = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first = False)
        
    def forward(self, omic, histo):
        #Nx32
        fused_vec = torch.cat([omic, histo], dim = 1 )#Nx64
        fused_vec= fused_vec.unsqueeze(0)
      
        x, _ = self.head(fused_vec, fused_vec, fused_vec)
        x = x.squeeze(0)
        return x
        
        
    
# class CrossAttentionBlock(nn.Module):
#     def __init__(self, dim = 64, hidden_dim = 64):
#         super(CrossAttentionBlock, self).__init__()
#         self.dim = dim
#         self.hidden_dim = hidden_dim
#         self.layer_norm1 = nn.LayerNorm(dim)
#         self.layer_norm2 = nn.LayerNorm(dim)
#         self.feed_forward = nn.Sequential(
#             nn.Linear(dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, dim)
#         )

#     def forward(self, query, key):
#         # Normalize input
#         query_norm = self.layer_norm1(query)
#         key_norm = self.layer_norm1(key)
#         value_norm = self.layer_norm1(key)

#         # Compute the dot product between query and key
#         scores = torch.matmul(query_norm, key_norm.transpose(-2, -1))
#         weights = F.softmax(scores, dim=-1)

#         # Apply attention to value
#         attention_output = torch.matmul(weights, value_norm)

#         # Second layer normalization
#         attention_output_norm = self.layer_norm2(attention_output)

#         # Feed-forward network
#         output = self.feed_forward(attention_output_norm)

#         return output 
    
# class SelfAttentionFusion(nn.Module):
#     def __init__(self, feature_dim = 64):
#         super(SelfAttentionFusion, self).__init__()
#         self.scale = 1.0 / (feature_dim ** 0.5)
#         self.query = nn.Linear(feature_dim, feature_dim)
#         self.key = nn.Linear(feature_dim, feature_dim)
#         self.value = nn.Linear(feature_dim, feature_dim)
#         #self.output = nn.Linear(feature_dim, feature_dim)
#         self.output = nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU())

#     def forward(self, tensor1, tensor2):
#         # Concatenate tensors along the feature dimension
#         x = torch.cat((tensor1, tensor2), dim=1)  # Shape: [1, 64]
#         # Compute queries, keys, values
#         queries = self.query(x)
#         keys = self.key(x)
#         values = self.value(x)
#         # Scaled dot-product attention
#         attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
#         attention_probs = F.softmax(attention_scores, dim=-1)
#         attention_out = torch.matmul(attention_probs, values)
#         # Passing through the final output linear layer
#         output = self.output(attention_out)
#         return output