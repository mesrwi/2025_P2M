import torch
import torch.nn as nn
from functools import partial
from model.DSTformer import DSTformer

def load_backbone(args):
    if not(hasattr(args, "backbone")):
        args.backbone = 'DSTformer' # Default
    
    if args.backbone == 'DSTformer':
        model_backbone = DSTformer(dim_in=3, dim_out=3, dim_feat=args.dim_feat, dim_rep=args.dim_rep, 
                                   depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio, norm_layer=partial(nn.LayerNorm, eps=1e-6), 
                                   maxlen=args.maxlen, num_joints=args.num_joints)
    else:
        raise Exception("Undefined backbone type.")
    
    return model_backbone