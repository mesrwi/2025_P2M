import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import pytorch_lightning as pl
from torchmetrics.functional.regression import mean_absolute_percentage_error, symmetric_mean_absolute_percentage_error

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=80):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class TimeDistributed(nn.Module):
    # Takes any module and stacks the time dimension with the batch dimenison of inputs before applying the module
    # Insipired from https://keras.io/api/layers/recurrent_layers/time_distributed/
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module # Can be any layer we wish to apply like Linear, Conv etc
        self.batch_first = batch_first
        
    def forward(self, x):
        if len(x.size()) <= 2: # 1 sample x: [timesteps, in_channel]
            return self.module(x)
        
        # Squash samples and timesteps into a single axis
        x = x.view(x.size(0), x.size(1), -1)
        x_reshape = x.contiguous().view(-1, x.size(-1)) # [samples, timesteps, input_size]
        y = self.module(x_reshape)
        
        # we have to reshape Y
        if self.batch_first:
            y = y.contiguous().view(x.size(0), -1, y.size(-1)) # (samples, timesteps, output_size)
        else:
            y = y.view(-1, x.size(1), y.size(-1)) # (timesteps, samples, output_size)
            
        return y

class GlobalPoseEmbedder(nn.Module):
    def __init__(self, embedding_dim, input_len, num):
        super(GlobalPoseEmbedder, self).__init__()
        self.input_embedder = TimeDistributed(nn.Linear(51, embedding_dim)) ##
        self.pos_encoding = PositionalEncoding(embedding_dim, max_len=input_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, dropout=0.2, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
    
    def forward(self, input):
        if input.dim() <= 2: input = input.unsqueeze(dim=-1)
        emb = self.input_embedder(input)
        emb = self.pos_encoding(emb) # emb: (batch, seq, feature)
        emb = self.encoder(emb)
        
        return emb

class SubjectFeatureEmbedder(nn.Module):
    def __init__(self, embedding_dim):
        super(SubjectFeatureEmbedder, self).__init__()
        self.embedding_dim = embedding_dim
        self.height_embedding = nn.Linear(1, embedding_dim)
        self.weight_embedding = nn.Linear(1, embedding_dim)
        self.muscle_embedding = nn.Linear(1, embedding_dim)
        self.fat_embedding = nn.Linear(1, embedding_dim)
        self.limb_embedding = nn.Linear(2, embedding_dim) # arm length, leg length
        
        self.fusion_layer = nn.Linear(embedding_dim * 5, embedding_dim)
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, subject_features):
        h, w, m, f, l = subject_features[:, 1].unsqueeze(1), subject_features[:, 2].unsqueeze(1), \
            subject_features[:, 3].unsqueeze(1), subject_features[:, 4].unsqueeze(1), subject_features[:, 5:7]
        h_emb, w_emb, m_emb, f_emb, l_emb = self.height_embedding(h), self.weight_embedding(w), self.muscle_embedding(m), self.fat_embedding(f), self.limb_embedding(l)
        subject_embeddings = self.fusion_layer(torch.cat([h_emb, w_emb, m_emb, f_emb, l_emb], dim=1))
        subject_embeddings = self.dropout(subject_embeddings)
        
        return subject_embeddings

class ImageEmbedder(nn.Module):
    def __init__(self):
        super(ImageEmbedder, self).__init__()
        # Img feature extraction
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        for p in self.resnet.parameters():
            p.requires_grad = False
        
        # Fine tune resnet
        # for c in list(self.resnet.children())[6:]:
        #     for p in c.parameters():
        #         p.requires_grad = True
        
    def forward(self, images):
        img_embeddings = self.resnet(images)
        size = img_embeddings.size()
        out = img_embeddings.view(*size[:2], -1)
        
        return out.view(*size).contiguous() # [batch_size, 2048, image_size/32, image_size/32]

class FeatureFusionNetwork(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, dropout=0.2):
        super(FeatureFusionNetwork, self).__init__()
        
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_linear = nn.Linear(2048, embedding_dim)
        input_dim = embedding_dim * 2 # image, subject
        self.feature_fusion = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.Linear(input_dim, input_dim, bias=False),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim, hidden_dim)
        )
    
    def forward(self, img_embedding, subject_embedding):
        features = torch.cat([img_embedding, subject_embedding], dim=1)
        features = self.feature_fusion(features)
        
        return features

class TransformerDecoderLayer(nn.Module):
    # custom decoder layer for non-autoregressive setting
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation='relu'):
        super(TransformerDecoderLayer, self).__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout)
        
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        
        self.activation = F.relu
        
    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super(TransformerDecoderLayer, self).__setstate__(state)
        
    def forward(self, tgt, memory, tgt_mask = None, memory_mask = None, tgt_key_padding_mask = None, 
        memory_key_padding_mask = None, tgt_is_causal = False, memory_is_causal = False):
        
        tgt2, attn_weights = self.multihead_attn(tgt, memory, memory)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt)))) # [1, b, d] -> [1, b, d_f] -> [1, b, d]
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, attn_weights

class Pose2Muscle(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.hid_dim = 64 # args['hidden_dim']
        self.emb_dim = 32 # args['embedding_dim']
        self.out_len = 80 # args['output_dim']
        self.gpu_num = 1 # args['gpu_num']
        self.lr = args['lr']
        self.save_hyperparameters()
        
        # Encoder
        self.tab_encoder = SubjectFeatureEmbedder(self.emb_dim)
        # self.img_encoder = ImageEmbedder() # resnet input size 맞춰야됨
        # self.txt_encoder = None
        # self.static_feature_encoder = FeatureFusionNetwork(self.emb_dim, self.hid_dim) # fusion net
        self.pose_encoder = GlobalPoseEmbedder(self.emb_dim, 80, None)
        
        # Decoder
        self.decoder_linear = None
        decoder_layer = TransformerDecoderLayer(self.emb_dim, nhead=4)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(self.emb_dim, self.out_len * len(args['target_muscle'])),
            nn.Dropout(0.2)
        )
        
        # for validation
        self.validation_step_outputs = []
    
    def forward(self, image, subject_features, pose3d):
        pose_embedding = self.pose_encoder(pose3d)
        
        subject_embedding = self.tab_encoder(subject_features)
        # img_embedding = self.img_encoder(image)
        # fusion_embedding = self.static_feature_encoder(img_embedding, subject_embedding)
        
        # tgt = fusion_embedding.unsqueeze(0)
        tgt = subject_embedding.unsqueeze(0) # tgt: [1, batch_size, d_model]
        memory = pose_embedding.permute(1, 0, 2) # memory: [seq_len, batch_size, d_model]
        decoder_out, attn_weights = self.decoder(tgt, memory)
        forecast = self.decoder_fc(decoder_out) # (1, b, d_model) -> (1, b, seq_len)
        
        return forecast.view(forecast.size()[1], self.out_len, -1), attn_weights
    
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def training_step(self, train_batch, batch_idx):
        filename, subject, pose3d, emg_values = train_batch
        image = torch.ones((224, 224))
        predicted_emg, _ = self.forward(image, subject, pose3d)
        loss = F.mse_loss(emg_values, predicted_emg)
        self.log('train_loss', loss)
        
        return loss
    
    def validation_step(self, test_batch, batch_idx):
        filename, subject, pose3d, emg_values = test_batch
        image = torch.ones((224, 224))
        predicted_emg, _ = self.forward(image, subject, pose3d)
        
        self.validation_step_outputs.append((emg_values, predicted_emg))
        
        return emg_values, predicted_emg
    
    def on_validation_epoch_end(self):
        emg_values, predicted_emg = [x[0] for x in self.validation_step_outputs], [x[1] for x in self.validation_step_outputs]
        emg_values, predicted_emg = torch.cat(emg_values), torch.cat(predicted_emg)
        # rescaled_emg, rescaled_predicted_emg = emg_values * scalar, predicted_emg * scalar
        loss = F.mse_loss(emg_values, predicted_emg)
        mae = F.l1_loss(emg_values, predicted_emg)
        mape = mean_absolute_percentage_error(predicted_emg, emg_values)
        smape = symmetric_mean_absolute_percentage_error(predicted_emg, emg_values)
        self.log('val_mae', mae)
        self.log('val_loss', loss)
        self.log('val_mape', mape)
        self.log('val_smape', smape)
        
        print('Validation MAE:', mae.detach().cpu().numpy(), 'LR:', self.optimizers().param_groups[0]['lr'])
        
        self.validation_step_outputs.clear()