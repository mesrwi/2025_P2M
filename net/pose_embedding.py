import torch.nn as nn
from _utils._utils import get_activation

class PoseEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding_dim = args['pose_embedding_dim']
        self.input_dim = 17 * 3
        self.window_size = 9 # args['window_size']
        self.half_window = self.window_size // 2
        
        self.conv = nn.Conv2d(in_channels=1, 
                              out_channels=self.embedding_dim, 
                              kernel_size=(51, self.window_size), 
                              stride=(1, 1), 
                              padding=(0, self.half_window))
        
        self.norm = nn.BatchNorm1d(self.embedding_dim)
        self.activation = get_activation(args['pose_embedding_activation'])
        
    def forward(self, x_keypoints, mask):
        '''
        keypoints: (B, T, J*3) - keypoint sequence (batch_size, seq_len, 17*3)
        mask: (B, T) - 1 for padding, otherwise 0
        '''
        B, T, D = x_keypoints.shape
        x_keypoints = x_keypoints.unsqueeze(1).permute(0, 1, 3, 2) # (B, 1, T, D) -> (B, 1, D, T)
        
        x = self.conv(x_keypoints) # (B, out_channel, 1, T)
        x = x.squeeze(2) # (B, out_channel, T)
        x = self.norm(x)
        x = self.activation(x)
        
        mask = mask.unsqueeze(1) # (B, 1, T)
        x = x * mask

        return x.permute(0, 2, 1)

    def __repr__(self):
        return "%s(embedding_dim=%d, input_dim=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.input_dim,
        )