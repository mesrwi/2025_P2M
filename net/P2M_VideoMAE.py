from transformers import VideoMAEModel, VideoMAEFeatureExtractor
import torch
import torch.nn as nn

import pytorch_lightning as pl
import torch.nn.functional as F
import torch
import torch.nn as nn
from torchmetrics import MeanAbsoluteError, MeanAbsolutePercentageError
from sklearn.metrics import r2_score
import numpy as np
import wandb

class VideoMAERegressor(nn.Module):
    def __init__(self, num_outputs=8, pretrained_model="MCG-NJU/videomae-base-finetuned-kinetics"):
        super().__init__()
        self.backbone = VideoMAEModel.from_pretrained(pretrained_model)

        # Backbone 출력: (B, T', hidden_dim) → 여기서 T'는 시간 축 다운샘플링된 값
        hidden_dim = self.backbone.config.hidden_size

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_outputs)  # 프레임별 회귀값
        )

    def forward(self, video):
        """
        video: (B, 3, T, H, W)
        return: (B, T', num_outputs)
        """
        outputs = self.backbone(video)
        features = outputs.last_hidden_state  # (B, T', hidden_dim)
        return self.regressor(features)  # (B, T', num_outputs)

class VideoMAERegressionModule(pl.LightningModule):
    def __init__(self, model: nn.Module, lr=1e-4):
        super().__init__()
        self.model = VideoMAERegressor()
        self.lr = lr
        self.criterion = nn.MSELoss()
        self.mae = MeanAbsoluteError()
        self.mape = MeanAbsolutePercentageError()
        self.all_preds = []
        self.all_targets = []
        self.sample_dirs = []
        self.example_input = None

    def forward(self, video):
        # if video.shape[1] == 3:
        #     # Already (B, 3, T, H, W)
        #     pass
        # else:
        #     # Convert from (B, T, 3, H, W)
        #     video = video.permute(0, 2, 1, 3, 4)
            
        return self.model(video)

    def training_step(self, batch, batch_idx):
        video, target, sample_dir = batch  # (B, 3, T, H, W), (B, T, C)
        pred = self(video)
        loss = self.criterion(pred, target)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        video, target, sample_dir = batch
        pred = self(video)
        self.log("val_mse", self.criterion(pred, target), prog_bar=True)
        self.log("val_mae", self.mae(pred, target), prog_bar=True)
        self.log("val_mape", self.mape(pred, target), prog_bar=True)

    def test_step(self, batch, batch_idx):
        video, target, sample_dir = batch
        pred = self(video)
        self.all_preds.append(pred.cpu())
        self.all_targets.append(target.cpu())
        self.sample_dirs += list(sample_dir)

        if self.example_input is None:
            self.example_input = {
                "gt": target[0].cpu(),
                "pred": pred[0].cpu()
            }

    def on_test_epoch_end(self):
        preds = torch.cat(self.all_preds)
        targets = torch.cat(self.all_targets)
        self.log_samplewise_r2_and_plot(preds, targets, self.sample_dirs)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def log_samplewise_r2_and_plot(self, preds, targets, sample_dirs=None, num_samples_to_log=30, num_worst_to_log=10):
        preds = preds.detach().cpu().numpy()
        targets = targets.detach().cpu().numpy()

        if preds.ndim == 2:
            preds = preds[:, :, np.newaxis]
            targets = targets[:, :, np.newaxis]

        N, T, C = preds.shape
        sample_dirs = sample_dirs or [f"Sample-{i}" for i in range(N)]

        for c in range(C):
            pred_c = preds[:, :, c]
            target_c = targets[:, :, c]
            samplewise_r2 = [r2_score(target_c[i], pred_c[i]) for i in range(N)]
            mean_r2 = np.mean(samplewise_r2)
            self.log(f"test_r2_ch{c}", mean_r2)

            if wandb.run:
                wandb.log({f"samplewise_r2_ch{c}": wandb.Histogram(samplewise_r2)})

                # Top R²
                top_images = []
                top_indices = np.argsort(samplewise_r2)[::-1][:num_samples_to_log]
                for i in top_indices:
                    fig = self.plot_prediction(pred_c[i], target_c[i], sample_dirs[i], r2=samplewise_r2[i])
                    top_images.append(wandb.Image(fig))
                wandb.log({f"top_predictions_ch{c}": top_images})

    def plot_prediction(self, pred, target, title, r2=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot(target, label='GT')
        ax.plot(pred, label='Pred')
        if r2:
            ax.set_title(f"{title} | R²={r2:.3f}")
        else:
            ax.set_title(title)
        ax.legend()
        return fig
