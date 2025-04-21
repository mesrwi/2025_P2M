import math
import numpy as np
from einops import rearrange, repeat
import torch
import torch.nn as nn
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import wandb
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from torchmetrics.regression import MeanAbsoluteError, MeanAbsolutePercentageError, SymmetricMeanAbsolutePercentageError, MeanSquaredError

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

class BaseP2MModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        # model
        self.args = args
        self.input_size = args.base.num_joints * args.base.dim_joints
        self.dim_feat = args.model.dim_feat
        self.temporal_range = args.model.temporal_range
        self.dim_model = args.model.dim_model
        
        # training
        self.learning_rate = args.training.learning_rate
        self.criterion = nn.MSELoss()
        
        # evaluation
        self.rmse = MeanSquaredError(squared=False)
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        self.smape = SymmetricMeanAbsolutePercentageError()
        self.all_preds, self.all_targets = [], []
        self.example_input = None
        
        self.sample_dirs = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def log_samplewise_r2_and_plot(self, preds, targets, sample_dirs=None, num_samples_to_log=30, num_worst_to_log=10):
        preds = preds.detach().cpu()
        targets = targets.detach().cpu()

        # Check shape and reshape if necessary
        if preds.ndim == 2:  # (N, T)
            preds = preds.unsqueeze(-1)  # (N, T, 1)
            targets = targets.unsqueeze(-1)
        elif preds.ndim == 3:
            pass  # Already (N, T, C)
        else:
            raise ValueError(f"Unexpected preds shape: {preds.shape}")

        N, T, C = preds.shape
        sample_dirs = sample_dirs or [f"Sample-{i}" for i in range(N)]

        mean_r2s = []
        all_top_images = []
        all_worst_images = []
        all_histograms = []

        for c in range(C):
            pred_c = preds[:, :, c].numpy()
            target_c = targets[:, :, c].numpy()
            
            samplewise_rmse = [mean_squared_error(target_c[i], pred_c[i], squared=False) for i in range(N)]
            samplewise_mae = [mean_absolute_error(target_c[i], pred_c[i]) for i in range(N)]
            samplewise_mape = [mean_absolute_percentage_error(target_c[i], pred_c[i]) for i in range(N)]
            samplewise_r2 = [r2_score(target_c[i], pred_c[i]) for i in range(N)]
            
            samplewise_smape = [self.smape(torch.from_numpy(pred_c[i]), torch.from_numpy(target_c[i])).item() for i in range(N)]
            
            mean_rmse = np.mean(samplewise_rmse)
            mean_mae = np.mean(samplewise_mae)
            mean_mape = np.mean(samplewise_mape)
            mean_smape = np.mean(samplewise_smape)
            mean_r2 = np.mean(samplewise_r2)
            
            mean_r2s.append(mean_r2)

            self.log(f"test_rmse_mean_samplewise_ch{c}", mean_rmse)
            self.log(f"test_mae_mean_samplewise_ch{c}", mean_mae)
            self.log(f"test_mape_mean_samplewise_ch{c}", mean_mape)
            self.log(f"test_smape_mean_samplewise_ch{c}", mean_smape)
            self.log(f"test_r2_mean_samplewise_ch{c}", mean_r2)

            if wandb.run is not None:
                # Histogram
                wandb.log({f"samplewise_r2_distribution_ch{c}": wandb.Histogram(samplewise_r2)})

                # Top R² 샘플 시각화
                top_images = []
                top_indices = np.argsort(samplewise_r2)[::-1][:num_samples_to_log]
                for rank, i in enumerate(top_indices):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(target_c[i], label="GT")
                    ax.plot(pred_c[i], label="Pred")
                    ax.set_title(f"HIGH R² | Ch {c} | #{rank+1}: {sample_dirs[i]}")
                    
                    metrics_text = f"R²: {samplewise_r2[i]:.4f}\nRMSE: {samplewise_rmse[i]:.4f}\nMAE: {samplewise_mae[i]:.4f}\nMAPE: {samplewise_mape[i]:.4f}\nSMAPE: {samplewise_smape[i]:.4f}"
                    ax.text(0.99, 0.95, metrics_text,
                            horizontalalignment='right',
                            verticalalignment='top',
                            transform=plt.gca().transAxes,
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
                    
                    # ax.legend()
                    top_images.append(wandb.Image(fig, caption=f"[Top-{rank+1}] Ch{c} {sample_dirs[i]}"))
                    plt.close(fig)

                # Worst R² 시각화
                worst_images = []
                sorted_indices = np.argsort(samplewise_r2)[:num_worst_to_log]
                for rank, i in enumerate(sorted_indices):
                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(target_c[i], label="GT")
                    ax.plot(pred_c[i], label="Pred")
                    ax.set_title(f"LOW R² | Ch {c} | #{rank+1}: {sample_dirs[i]}")
                    
                    metrics_text = f"R²: {samplewise_r2[i]:.4f}\nRMSE: {samplewise_rmse[i]:.4f}\nMAE: {samplewise_mae[i]:.4f}\nMAPE: {samplewise_mape[i]:.4f}\nSMAPE: {samplewise_smape[i]:.4f}"
                    ax.text(0.99, 0.95, metrics_text,
                            horizontalalignment='right',
                            verticalalignment='top',
                            transform=plt.gca().transAxes,
                            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5'))
                    
                    # ax.legend()
                    worst_images.append(wandb.Image(fig, caption=f"[Worst-{rank+1}] Ch{c} {sample_dirs[i]}"))
                    plt.close(fig)

                all_top_images.extend(top_images)
                all_worst_images.extend(worst_images)

        # 최종 wandb에 로깅
        if wandb.run is not None:
            if all_top_images:
                wandb.log({"sample_predictions_top": all_top_images})
            if all_worst_images:
                wandb.log({"sample_predictions_worst": all_worst_images})

class P2M(BaseP2MModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        # Spatial & Local temporal encoder
        self.joints_embed = nn.Linear(self.input_size, self.dim_feat)
        if self.args.model.use_cond:
            self.conv = nn.Conv2d(1,  self.dim_feat, 
                                (self.input_size, self.temporal_range), 
                                (1, 1), 
                                (0, self.temporal_range // 2))
            self.positional_encoder = PositionalEncoding(dim_model=self.dim_feat)
        else:
            self.conv = nn.Conv2d(1,  self.dim_model, 
                                (self.input_size, self.temporal_range), 
                                (1, 1), 
                                (0, self.temporal_range // 2))
            self.positional_encoder = PositionalEncoding(dim_model=self.dim_model)
        # Global encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=self.dim_model, nhead=self.args.model.num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=self.args.model.num_layers)
        
        # Regression head
        self.regression_head = nn.ModuleList([nn.Linear(self.dim_model, 1) for _ in range(len(self.args.muscles))]) # nn.Linear(self.dim_model, self.args.model.output_dim)
    
    def forward(self, x, condval):
        '''
        x: [batch_size, seq_len, n_joints, 3]
        condval: [batch_size, ]
        '''
        B, T, N = x.size()[:3]
        
        x = rearrange(x, 'b t n c -> b 1 (c n) t', c=3)
        
        # conv(x): [b, 1, 3n, t] -> [b, dim, 1, t] -> [b, t, dim]
        conv_out = rearrange(self.conv(x).squeeze(2), 'b c t -> b t c')
        src = self.positional_encoder(conv_out)
        
        if self.args.model.use_cond:
            condition = repeat(condval, 'b -> b t 2', t=T)
            transformer_out = self.transformer(torch.cat([src, condition], dim=2))
        else:
            transformer_out = self.transformer(src)
            
        out = [head(transformer_out) for head in self.regression_head]
        out = torch.cat(out, dim=2)  # (B, T, C)
        
        return out
    
    def training_step(self, batch, batch_idx):
        image, pose, emg, condval, sample_dir = batch
        pred_emg = self(pose, condval)
        loss = self.criterion(pred_emg, emg)
        self.log("train_loss(MSE)", loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, pose, emg, condval, sample_dir = batch
        pred_emg = self(pose, condval)
        
        self.log("val_mse", self.criterion(pred_emg, emg), prog_bar=True, sync_dist=True)
        
        self.log("val_mae", self.mae(pred_emg, emg), prog_bar=True, sync_dist=True)
        self.log("val_mape", self.mape(pred_emg, emg), prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        imgae, pose, emg, condval, sample_dir = batch
        pred_emg = self(pose, condval)
        
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