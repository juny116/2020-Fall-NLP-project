from torch import nn
from torch.nn import init
import torch

class BahdanauAttention(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.W1 = nn.Linear(in_features=2 * hidden_dim, out_features=2 * hidden_dim)
        self.W2 = nn.Linear(in_features=2 * hidden_dim, out_features=2 * hidden_dim)
        self.V = nn.Linear(in_features=2 * hidden_dim, out_features=1)
        self.tanh = nn.Tanh()
        

    def forward(self, rnn_out, state):
        merged_state = torch.cat([s for s in state[-1]],1)
        merged_state = merged_state.squeeze(0).unsqueeze(1)   
        score = self.V(self.tanh(self.W1(rnn_out.permute(1, 0, 2)) + self.W2(merged_state)))
        weights = torch.nn.functional.softmax(score, dim=1)
        result = torch.bmm(rnn_out.permute(1, 2, 0), weights)
        return result