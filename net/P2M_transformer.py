import math
import torch
import torch.nn as nn
from .STGCN import STGCNEncoder, get_h36m_adjacency
from .legacy_model import BaseP2MModel

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

class P2MTransformer(BaseP2MModel):
    def __init__(self, args):
        super().__init__(args)
        
        A = get_h36m_adjacency()
        # skeleton graph encoder
        self.stgcn = STGCNEncoder(in_channels=args.base.dim_joints, num_joints=args.base.num_joints, A=A)
        self.input_proj = nn.Linear(args.base.num_joints * 256, args.model.dim_model)
        
        # image encoder: Noen -> skeleton-only
        self.image_encoder = None
        
        # transformer encoder for global information
        self.pos_encoder = PositionalEncoding(dim_model=args.model.dim_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.model.dim_model, nhead=args.model.num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.model.num_layers)
        
        self.output_proj = nn.Linear(args.model.dim_model, len(args.muscles))
        self.output_seq_len = args.base.clip_len
        self.num_channels = len(args.muscles)
        
    
    def forward(self, x):
        # x: (B, T, V, C) -> (B, C, T, V)
        x = x.permute(0, 3, 1, 2)
        x = self.stgcn(x)  # (B, C_out, T_out, V)
        x = x.permute(0, 2, 3, 1)  # (B, T_out, V, C_out)
        x = x.reshape(x.size(0), x.size(1), -1)  # (B, T_out, V*C_out)
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)  # (B, T_out, V*C_out)
        x = self.output_proj(x)  # (B, T_out, num_channels)
        
        return x # .permute(0, 2, 1)  # (B, num_channels, output_seq_len)

    def training_step(self, batch, batch_idx):
        pose, emg, condval, sample_dir = batch
        pred_emg = self(pose)
        loss = self.criterion(pred_emg, emg)
        self.log("train_loss(MSE)", loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        pose, emg, condval, sample_dir = batch
        pred_emg = self(pose)
        
        self.log("val_mse", self.criterion(pred_emg, emg), prog_bar=True, sync_dist=True)
        self.log("val_rmse", self.rmse(pred_emg, emg), prog_bar=True, sync_dist=True)
        self.log("val_mae", self.mae(pred_emg, emg), prog_bar=True, sync_dist=True)
        self.log("val_mape", self.mape(pred_emg, emg), prog_bar=True, sync_dist=True)
        self.log("val_smape", self.mape(pred_emg, emg), prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        pose, emg, condval, sample_dir = batch
        pred_emg = self(pose)
        
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