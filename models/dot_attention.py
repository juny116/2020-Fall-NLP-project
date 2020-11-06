from torch import nn
from torch.nn import init
import torch

class DotAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.linear = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        

    def forward(self, rnn_out, state):
        merged_state = torch.cat([s for s in state[-1]],1)
        merged_state = merged_state.squeeze(0).unsqueeze(2)
        weights = torch.bmm(rnn_out.permute(1, 0, 2), merged_state)
        weights = torch.nn.functional.softmax(weights.squeeze(2)).unsqueeze(2)
        result = torch.bmm(rnn_out.permute(1, 2, 0), weights)
        return result