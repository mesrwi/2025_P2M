import torch
import torch.nn as nn
from torch import Tensor

from .transformer_layers import TransformerEncoderLayer, PositionalEncoding

class Encoder(nn.Module):
    @property
    def output_size(self):
        return self._output_size
    
class TransformerEncoder(Encoder):
    def __init__(self, args):
        super(TransformerEncoder, self).__init__()

        self.pe = PositionalEncoding(args['pose_embedding_dim'])
        self.emb_dropout = nn.Dropout(p=args['encoder_emb_dropout'])

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=args['encoder_dim'],
                    ff_size=args['encoder_ff_size'],
                    num_heads=args['encoder_nhead'],
                    dropout=args['encoder_emb_dropout'],
                )
                for _ in range(args['encoder_layer'])
            ]
        )

        self.layer_norm = nn.LayerNorm(args['encoder_dim'], eps=1e-6)
        
        self.output_layer = nn.Linear(args['encoder_dim'], args['encoder_dim'])
        self._output_size = args['encoder_dim']

    def forward(self, embed_src: Tensor, mask=None, sub=None) -> (Tensor):
        '''
        embed_src: (batch_size, seq_len, d_model)
        mask: (batch_size, seq_len)
        x: (seq_len, batch_size, d_model)
        '''
        embed_src = embed_src.permute(1, 0, 2)
        mask = mask.permute(1, 0)
        
        # positional embedding
        x = self.pe(embed_src, mask)
        x = self.emb_dropout(x)

        x = x.permute(1, 0, 2)
        mask = mask.permute(1, 0).unsqueeze(1)
        for layer in self.layers:
            x = layer(x, mask)
        
        encoder_output = self.layer_norm(x) # [batch_size, seq_len, dim]

        out = self.output_layer(encoder_output) # [batch_size, seq_len, gloss_vocab_size]
        
        return out