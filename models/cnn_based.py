import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence

class CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_filters, filter_sizes, num_classes, dropout_prob=0):
        super().__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(
                in_channels = 1,
                out_channels = n_filters, 
                kernel_size = (size, input_dim)
            )
            for size in filter_sizes
        ])

        self.fc1 = nn.Linear(len(filter_sizes) * n_filters, hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, num_classes)

        self.dropout = nn.Dropout(dropout_prob)
    
    def forward(self, text, length):
        text = text.transpose(0,1)
        text = text.unsqueeze(1)
        
        conved = [F.relu(conv(text)).squeeze(3) for conv in self.convs]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        dropped = self.dropout(torch.cat(pooled, dim = 1))

        output = F.relu(self.fc1(dropped))

        output = self.fc2(output)

        return output

class CNNClassifier(nn.Module):

    def __init__(self, num_classes, word_dim, hidden_dim, clf_dim, n_filters, filter_sizes, dropout_prob=0):
        super().__init__()
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.clf_dim = clf_dim
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(dropout_prob)

        self.cnn = CNN(
            input_dim=word_dim, 
            hidden_dim=hidden_dim, 
            n_filters = n_filters,
            filter_sizes = filter_sizes,
            num_classes = num_classes,
            dropout_prob=dropout_prob
            )

    def forward(self, inputs, length, batch_first=False):
        
        if batch_first:
            inputs = inputs.transpose(0, 1)
        inputs = self.dropout(inputs)
        
        logit = self.cnn(text=inputs, length = length)
        return logit
