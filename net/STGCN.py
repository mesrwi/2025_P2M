import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_h36m_adjacency():
    # 관절 간 연결 리스트 (zero-based indexing)
    edges = [
        (0, 1), (1, 2), (2, 3),       # 오른쪽 다리
        (0, 4), (4, 5), (5, 6),       # 왼쪽 다리
        (0, 7), (7, 8), (8, 9), (9, 10),  # 허리 → 머리
        (8, 14), (14, 15), (15, 16),  # 오른팔
        (8, 11), (11, 12), (12, 13)   # 왼팔
    ]
    V = 17
    A = np.eye(V)
    for i, j in edges:
        A[i, j] = 1
        A[j, i] = 1  # 무방향 그래프

    return torch.tensor(A, dtype=torch.float32)

class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True):
        super().__init__()
        self.A = A
        self.gcn = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(9, 1), padding=(4, 0), stride=(stride, 1)),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace=True)
        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        res = self.residual(x)  # Store residual before modifying x
        A = self.A.to(x.device)
        x_gcn = torch.einsum('nctv,vw->nctw', x, A)
        x = self.gcn(x_gcn)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)

class STGCNEncoder(nn.Module):
    def __init__(self, in_channels, num_joints, A):
        super().__init__()
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)
        self.layers = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, residual=False),
            STGCNBlock(64, 64, A),
            STGCNBlock(64, 128, A),
            STGCNBlock(128, 256, A)
        ])

    def forward(self, x):
        # x: (B, C, T, V)
        N, C, T, V = x.shape
        x = x.reshape(N, C * V, T)
        x = self.data_bn(x)
        x = x.reshape(N, C, T, V)
        for layer in self.layers:
            x = layer(x)
        return x  # (B, C_out, T_out, V)