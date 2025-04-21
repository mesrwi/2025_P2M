import math
import torch
import torch.nn as nn
from .STGCN import STGCNEncoder, get_h36m_adjacency
from .legacy_model import BaseP2MModel

from transformers import CLIPModel, CLIPProcessor
from peft import get_peft_model, LoraConfig, TaskType

def get_lora_clip_model():
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    peft_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]  # vision transformer 기준
    )
    model.vision_model = get_peft_model(model.vision_model, peft_config)
    return model

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

class P2MMultimodalCLIP(BaseP2MModel):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.dim_model = args.model.dim_model

        # 1. Pose Encoder
        A = get_h36m_adjacency()
        self.stgcn = STGCNEncoder(
            in_channels=args.base.dim_joints,
            num_joints=args.base.num_joints,
            A=A
        )
        self.pose_proj = nn.Linear(args.base.num_joints * 256, self.dim_model)

        # 2. CLIP (with LoRA on vision encoder)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        vision_peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=["q_proj", "v_proj", "k_proj", "out_proj"]
        )
        self.clip_model.vision_model = get_peft_model(
            self.clip_model.vision_model, vision_peft_config
        )
        self.tokenizer = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32").tokenizer

        # 3. Cross-Attention: image(query), pose(key/value)
        self.cross_attn = nn.MultiheadAttention(embed_dim=self.dim_model, num_heads=8, batch_first=True)

        # 4. Transformer
        self.pos_encoder = PositionalEncoding(dim_model=self.dim_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.dim_model,
            nhead=args.model.num_heads,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.model.num_layers)

        # 5. Output
        if self.dim_model == 512:
            self.output_proj = nn.Linear(self.dim_model, len(args.muscles))
        elif self.dim_model == 2048:
            self.output_proj = nn.Sequential(
                nn.Linear(self.dim_model, 512),
                nn.ReLU(),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Linear(128, len(args.muscles))
            )

    def forward(self, pose, rgb, text_seq):
        """
        pose: (B, T, V, C)
        rgb: (B, T, 3, 224, 224)
        text_seq: List[List[str]] (B, T)
        """
        B, T, _, _, _ = rgb.shape

        # 1. Pose → ST-GCN
        pose = pose.permute(0, 3, 1, 2)  # (B, C, T, V)
        pose = self.stgcn(pose)  # (B, C_out, T, V)
        pose = pose.permute(0, 2, 3, 1).reshape(B, T, -1)  # (B, T, V*C)
        pose = self.pose_proj(pose)  # (B, T, D)

        # 2. Image → CLIP Vision
        rgb = rgb.view(B * T, 3, 224, 224)
        with torch.no_grad():
            vis_out = self.clip_model.vision_model(rgb)['last_hidden_state'][:, 0, :]
        img_embed = self.clip_model.visual_projection(vis_out).view(B, T, -1)  # (B, T, D)

        # 3. Text → CLIP Text
        # Flatten text into B*T list
        flat_texts = [txt for seq in text_seq for txt in seq]
        text_inputs = self.tokenizer(flat_texts, return_tensors="pt", padding=True, truncation=True)
        text_inputs = {k: v.to(rgb.device) for k, v in text_inputs.items()}
        with torch.no_grad():
            text_out = self.clip_model.text_model(**text_inputs).last_hidden_state[:, 0, :]
        text_embed = self.clip_model.text_projection(text_out).view(B, T, -1)  # (B, T, D)

        # 4. Cross-Attn: Query = Image, Key/Value = Pose
        fused, _ = self.cross_attn(query=img_embed, key=pose, value=pose)  # (B, T, D)

        # 5. Add Text Embedding
        fused = fused + text_embed  # (B, T, D)

        # 6. Transformer + Output
        x = self.pos_encoder(fused)
        x = self.transformer(x)
        out = self.output_proj(x)  # (B, T, num_muscles)
        return out
    
    def training_step(self, batch, batch_idx):
        image, pose, emg, condval, text, sample_dir = batch
        pred_emg = self(pose, image, text)
        loss = self.criterion(pred_emg, emg)
        self.log("train_loss(MSE)", loss, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, pose, emg, condval, text, sample_dir = batch
        pred_emg = self(pose, image, text)
        
        self.log("val_mse", self.criterion(pred_emg, emg), prog_bar=True, sync_dist=True)
        self.log("val_mae", self.mae(pred_emg, emg), prog_bar=True, sync_dist=True)
        self.log("val_mape", self.mape(pred_emg, emg), prog_bar=True, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        image, pose, emg, condval, text, sample_dir = batch
        pred_emg = self(pose, image, text)
        
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