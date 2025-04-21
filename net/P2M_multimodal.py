import math
import torch
import torch.nn as nn
from .STGCN import STGCNEncoder, get_h36m_adjacency
from .legacy_model import BaseP2MModel
from .ImageEncoder import ResNetEmbedder

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model: int, dropout_p: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)

        pe = torch.zeros(max_len, dim_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].to(x.device)
        return self.dropout(x)

class P2MMultimodal(BaseP2MModel):
    def __init__(self, args):
        super().__init__(args)
        
        A = get_h36m_adjacency()
        # skeleton graph encoder
        self.stgcn = STGCNEncoder(in_channels=args.base.dim_joints, num_joints=args.base.num_joints, A=A)
        self.input_proj = nn.Linear(args.base.num_joints * 256, args.model.dim_model)
        
        # image encoder
        # self.image_encoder = ResNetEmbedder(resnet_type='resnet34', pretrained=True).eval()
        self.cross_attn = nn.MultiheadAttention(embed_dim=args.model.dim_model, num_heads=8, batch_first=True)
        
        # transformer encoder for global information
        self.pos_encoder = PositionalEncoding(dim_model=args.model.dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.model.dim_model, nhead=args.model.num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.model.num_layers)
        
        if args.model.dim_model == 512:
            self.output_proj = nn.Linear(args.model.dim_model, len(args.muscles))
        elif args.model.dim_model == 2048:
            self.output_proj = nn.Sequential(
                nn.Linear(args.model.dim_model, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, len(args.muscles))
            )
        self.output_seq_len = args.base.clip_len
        self.num_channels = len(args.muscles)
        
    
    def forward(self, pose, rgb):
        '''
        pose: (B, T, V, C)
        rgb: (B, T, 3, 256, 256)
        '''
        # pose: (B, T, V, C) -> (B, C, T, V)
        pose = pose.permute(0, 3, 1, 2)
        pose = self.stgcn(pose)  # (B, C_out, T_out, V)
        pose = pose.permute(0, 2, 3, 1)  # (B, T_out, V, C_out)
        pose = pose.reshape(pose.size(0), pose.size(1), -1)  # (B, T_out, V*C_out)
        pose = self.input_proj(pose)
        
        # B, T, C, H, W = rgb.size()
        # rgb = rgb.reshape(B*T, C, H, W) # (B * T, 3, 224, 224)
        # rgb = self.image_encoder(rgb) # (B * T, 512)
        # rgb = rgb.reshape(B, T, -1)
        B, T, D = rgb.size()
        
        fused_rgb, _ = self.cross_attn(query=rgb, key=pose, value=pose) # (B, T, 512)
        
        # x = torch.cat([pose, rgb], dim=-1)
        x = self.pos_encoder(fused_rgb)
        x = self.transformer(x)  # (B, T_out, V*C_out)
        x = self.output_proj(x)  # (B, T_out, num_channels)
        
        return x # .permute(0, 2, 1)  # (B, num_channels, output_seq_len)

    def training_step(self, batch, batch_idx):
        image, pose, emg, condval, sample_dir = batch
        pred_emg = self(pose, image)
        loss = self.criterion(pred_emg, emg)
        self.log("train_loss(MSE)", loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, pose, emg, condval, sample_dir = batch
        pred_emg = self(pose, image)
        
        self.log("val_mse", self.criterion(pred_emg, emg), prog_bar=True, sync_dist=True)
        self.log("val_mae", self.mae(pred_emg, emg), prog_bar=True, sync_dist=True)
        self.log("val_mape", self.mape(pred_emg, emg), prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        image, pose, emg, condval, sample_dir = batch
        pred_emg = self(pose, image)
        
        self.all_preds.append(pred_emg.cpu())
        self.all_targets.append(emg.cpu())
        self.sample_dirs += list(sample_dir)  # sample_dir is a batch of strings
        
        if self.example_input is None:
            self.example_input = {
                "gt": emg[0].cpu(),
                "pred": pred_emg[0].cpu()
            }
    
    def on_test_epoch_end(self):
        self.log_samplewise_r2_and_plot(
            torch.cat(self.all_preds),
            torch.cat(self.all_targets),
            self.sample_dirs
        )

class P2MMultimodal_cross(BaseP2MModel):
    def __init__(self, args):
        super().__init__(args)
        
        A = get_h36m_adjacency()
        self.stgcn = STGCNEncoder(in_channels=args.base.dim_joints, num_joints=args.base.num_joints, A=A)
        self.input_proj = nn.Linear(args.base.num_joints * 256, args.model.dim_model)

        self.pos_encoder = PositionalEncoding(dim_model=args.model.dim_model)

        # 💡 self-attention encoders for pose and rgb
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.model.dim_model, nhead=args.model.num_heads, batch_first=True)
        self.pose_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.model.num_layers)
        self.rgb_encoder = nn.TransformerEncoder(encoder_layer, num_layers=args.model.num_layers)

        # 💡 cross-attention
        self.cross_attn = nn.MultiheadAttention(embed_dim=args.model.dim_model, num_heads=8, batch_first=True)

        # output projection
        if args.model.dim_model == 512:
            self.output_proj = nn.Linear(args.model.dim_model, len(args.muscles))
        elif args.model.dim_model == 2048:
            self.output_proj = nn.Sequential(
                nn.Linear(args.model.dim_model, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, len(args.muscles))
            )

        self.output_seq_len = args.base.clip_len
        self.num_channels = len(args.muscles)
        
    
    def forward(self, pose, rgb):
        '''
        pose: (B, T, V, C)
        rgb: (B, T, D)  # 이미 임베딩된 상태라고 가정
        '''
        B, T, D = rgb.size()

        # pose processing
        pose = pose.permute(0, 3, 1, 2)               # (B, C, T, V)
        pose = self.stgcn(pose)                      # (B, C_out, T_out, V)
        pose = pose.permute(0, 2, 3, 1)               # (B, T_out, V, C_out)
        pose = pose.reshape(B, T, -1)                 # (B, T_out, V*C_out)
        pose = self.input_proj(pose)                 # (B, T, dim_model)

        # positional encoding
        pose = self.pos_encoder(pose)
        rgb = self.pos_encoder(rgb)

        # 🧠 apply self-attention separately
        pose_feat = self.pose_encoder(pose)          # (B, T, dim_model)
        rgb_feat = self.rgb_encoder(rgb)             # (B, T, dim_model)

        # 🔁 cross attention: query = rgb, key/value = pose
        fused, _ = self.cross_attn(query=rgb_feat, key=pose_feat, value=pose_feat)  # (B, T, dim_model)

        # 🔚 project to output
        out = self.output_proj(fused)  # (B, T, num_channels)
        return out

    def training_step(self, batch, batch_idx):
        image, pose, emg, condval, sample_dir = batch
        pred_emg = self(pose, image)
        loss = self.criterion(pred_emg, emg)
        self.log("train_loss(MSE)", loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, pose, emg, condval, sample_dir = batch
        pred_emg = self(pose, image)
        
        self.log("val_mse", self.criterion(pred_emg, emg), prog_bar=True, sync_dist=True)
        self.log("val_mae", self.mae(pred_emg, emg), prog_bar=True, sync_dist=True)
        self.log("val_mape", self.mape(pred_emg, emg), prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        image, pose, emg, condval, sample_dir = batch
        pred_emg = self(pose, image)
        
        self.all_preds.append(pred_emg.cpu())
        self.all_targets.append(emg.cpu())
        self.sample_dirs += list(sample_dir)  # sample_dir is a batch of strings
        
        if self.example_input is None:
            self.example_input = {
                "gt": emg[0].cpu(),
                "pred": pred_emg[0].cpu()
            }
    
    def on_test_epoch_end(self):
        self.log_samplewise_r2_and_plot(
            torch.cat(self.all_preds),
            torch.cat(self.all_targets),
            self.sample_dirs
        )