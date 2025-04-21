import torch.nn as nn
from _utils._utils import get_activation

class SubjectEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.embedding_dim = args['subject_embedding_dim']
        self.input_dim = 7
        
        self.lin1 = nn.Linear(self.input_dim, self.embedding_dim)
        self.lin2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        
        self.norm1 = nn.BatchNorm1d(self.embedding_dim)
        self.norm2 = nn.BatchNorm1d(self.embedding_dim)
        self.activation = get_activation(args['subject_embedding_activation'])
        
    def forward(self, x):
        out = self.lin1(x)
        out = self.activation(self.norm1(out))
        
        out = self.norm2(self.lin2(out))

        return out

    def __repr__(self):
        return "%s(embedding_dim=%d, input_dim=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.input_dim,
        )