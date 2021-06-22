
import torch
import torch.nn as nn
from torch.nn import functional as F

from transformers import AutoModel

class EntityTagger(nn.Module):
    def __init__(self, params):
        super(EntityTagger, self).__init__()
        self.num_tag = params.num_tag
        self.hidden_dim = params.hidden_dim
        self.model = AutoModel.from_pretrained(params.model_name)
        self.dropout = nn.Dropout(params.dropout)

        self.linear = nn.Linear(self.hidden_dim, self.num_tag)

    def forward(self, X):
        outputs = self.model(X) # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        outputs = outputs[0] # (bsz, seq_len, hidden_dim)
        
        outputs = self.dropout(outputs)
        prediction = self.linear(outputs)

        return prediction
