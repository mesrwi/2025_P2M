import math
import warnings
import torch
import torch.nn as nn
from collections import OrderedDict
from model.drop import DropPath

def _no_grad_trunc_normal(tensor, mean, std, a, b):
    # 잘린 정규분포로 가중치 초기화
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.
    
    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                "The distribution of values may be incorrect.",
                stacklevel=2)
    
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        
        #Uniformly fill tensor with values from [l, u], then translate to [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        
        tensor.clamp_(min=a, max=b)
        
        return tensor

def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal(tensor, mean, std, a, b)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., st_mode='vanilla'):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim) # qkv 만드는 선형 레이어 아님
        self.mode = st_mode
        if self.mode == 'parallel':
            self.ts_attn = nn.Linear(dim * 2, dim * 2)
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        
        self.attn_count_s = None
        self.attn_count_t = None
    
    def forward(self, x, seqlen):
        B, N, C = x.shape
        if self.mode == 'spatial':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_spatial(q, k, v)
        elif self.mode == 'temporal':
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [3, B, num_heads, N, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
            x = self.forward_temporal(q, k, v, seqlen)
        else:
            raise NotImplementedError(self.mode)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
            
    def forward_spatial(self, q, k, v):
        B, _, N, C = q.shape
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C * self.num_heads)
        
        return x
    
    def forward_temporal(self, q, k, v, seqlen):
        B, _, N, C = q.shape
        qt = q.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4) # [B, H, N, T, C]
        kt = k.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4) # [B, H, N, T, C]
        vt = v.reshape(-1, seqlen, self.num_heads, N, C).permute(0, 2, 3, 1, 4) # [B, H, N, T, C]
        
        attn = (qt @ kt.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = attn @ vt
        x = x.permute(0, 3, 2, 1, 4).reshape(B, N, C*self.num_heads)
        
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., mlp_out_ratio=1., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, st_mode='stage_st', att_fuse=False):
        super().__init__()
        self.st_mode = st_mode
        self.norm1_s = norm_layer(dim)
        self.norm1_t = norm_layer(dim)
        self.attn_s = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, st_mode='spatial')
        self.attn_t = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, st_mode='temporal')
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2_s = norm_layer(dim)
        self.norm2_t = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        mlp_out_dim = int(dim * mlp_out_ratio)
        self.mlp_s = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer, drop=drop)
        self.mlp_t = MLP(in_features=dim, hidden_features=mlp_hidden_dim, out_features=mlp_out_dim, act_layer=act_layer, drop=drop)
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.Linear(dim * 2, dim * 2)
    
    def forward(self, x, seqlen_tensor):
        seqlen = seqlen_tensor.item()
        if self.st_mode == 'stage_st':
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
        elif self.st_mode == 'stage_ts':
            x = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x = x + self.drop_path(self.mlp_t(self.norm2_t(x)))
            x = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x = x + self.drop_path(self.mlp_s(self.norm2_s(x)))
        elif self.st_mode == 'stage_para':
            x_t = x + self.drop_path(self.attn_t(self.norm1_t(x), seqlen))
            x_t = x_t + self.drop_path(self.mlp_t(self.norm2_t(x_t)))
            x_s = x + self.drop_path(self.attn_s(self.norm1_s(x), seqlen))
            x_s = x_s + self.drop_path(self.mlp_s(self.norm2_s(x_s)))
            
            if self.att_fuse:
                # x_s, x_t: [BF, J, dim]
                alpha = torch.cat([x_s, x_t], dim=-1) # alpha: [BF, J, dim*2]
                BF, J = alpha.shape[:2]
                alpha = self.ts_attn(alpha).reshape(BF, J, -1, 2)
                alpha = alpha.softmax(dim=-1)
                x = x_s * alpha[:, :, :, 0] + x_t * alpha[:, :, :, 1]
            else:
                x = (x_s + x_t) * 0.5
        else:
            raise NotImplementedError(self.st_mode)
        
        return x

class DSTformer(nn.Module):
    def __init__(self, dim_in=3, dim_out=3, dim_feat=256, dim_rep=512,
                depth=5, num_heads=8, mlp_ratio=4,
                num_joints=17, maxlen=243,
                qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, att_fuse=True):
        super().__init__()
        self.dim_out = dim_out
        self.dim_feat = dim_feat
        self.joints_embed = nn.Linear(dim_in, dim_feat)
        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks_st = nn.ModuleList([
            Block(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode='stage_st') for i in range(depth)
            ]) # Spatial -> Temporal path
        self.blocks_ts =nn.ModuleList([
            Block(dim=dim_feat, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                st_mode='stage_ts') for i in range(depth)
            ]) # Temporal -> Spatial path
        self.norm = norm_layer(dim_feat)
        if dim_rep:
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(dim_feat, dim_rep)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()
        self.head = nn.Linear(dim_rep, dim_out) if dim_out > 0 else nn.Identity() # Regression head
        self.temp_embed = nn.Parameter(torch.zeros(1, maxlen, 1, dim_feat))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_joints, dim_feat))
        
        # PositionalEmbedding & model weight 초기화
        trunc_normal_(self.temp_embed, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)
        
        self.att_fuse = att_fuse
        if self.att_fuse:
            self.ts_attn = nn.ModuleList([nn.Linear(dim_feat * 2, 2) for i in range(depth)])
            for i in range(depth):
                self.ts_attn[i].weight.data.fill_(0)
                self.ts_attn[i].bias.data.fill_(0.5)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, x, return_rep=False):
        B, F, J, C = x.shape # [batch_size, num_frames, num_joints, dimension=3; (x, y, score)]
        x = x.reshape(-1, J, C)
        BF = x.shape[0]
        seqlen_tensor = torch.tensor(F).to(x.device)
        
        # High-dimensional feature of input 2D
        x = self.joints_embed(x) # 3 -> 256
        
        # Positional Embedding
        x = x + self.pos_embed # 각 관절(joint)에 대해 고유한 학습 가능한 위치 인코딩을 부여
        _, J, C = x.shape
        x = x.reshape(-1, F, J, C) + self.temp_embed[:, :F, :, :] # 각 프레임에 대한 위치 인코딩 부여
        x = x.reshape(BF, J, C)
        x = self.pos_drop(x)
        alphas = []
        
        # DSTformer
        for idx, (blk_st, blk_ts) in enumerate(zip(self.blocks_st, self.blocks_ts)):
            x_st = blk_st(x, seqlen_tensor)
            x_ts = blk_ts(x, seqlen_tensor)
            
            # 가중합을 위한 alpha
            if self.att_fuse:
                att = self.ts_attn[idx]
                alpha = torch.cat([x_st, x_ts], dim=-1)
                BF, J = alpha.shape[:2]
                alpha = att(alpha)
                alpha = alpha.softmax(dim=-1)
                x = x_st * alpha[:, :, 0:1] + x_ts * alpha[:, :, 1:2]
            else:
                x = (x_st + x_ts) * 0.5
        
        x = self.norm(x)
        x = x.reshape(B, F, J, -1)
        x = self.pre_logits(x) # [B, F, J, dim_rep]
        if return_rep:
            return x
        x = self.head(x)
        return x
    
    def get_representation(self, x):
        return self.foward(x, return_rep=True)