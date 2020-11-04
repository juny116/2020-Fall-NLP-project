from torch import nn
from torch.nn import init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiRNNMax(nn.Module):

    def __init__(self, rnn_type, input_dim, hidden_dim, dropout_prob=0):
        super().__init__()
        self.rnn_type = rnn_type
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout_prob = dropout_prob

        if rnn_type == 'gru':
            self.rnn = nn.GRU(
                input_size=input_dim, hidden_size=hidden_dim,
                bidirectional=True, dropout=dropout_prob, num_layers=2)
        elif rnn_type == 'lstm':
            self.rnn = nn.LSTM(
                input_size=input_dim, hidden_size=hidden_dim,
                bidirectional=True, dropout=dropout_prob, num_layers=2)
        else:
            raise ValueError('Unknown RNN type!')
        self.reset_parameters()

    def reset_parameters(self):
        init.orthogonal_(self.rnn.weight_hh_l0.data)
        init.kaiming_normal_(self.rnn.weight_ih_l0.data)
        init.constant_(self.rnn.bias_hh_l0.data, val=0)
        init.constant_(self.rnn.bias_ih_l0.data, val=0)
        init.orthogonal_(self.rnn.weight_hh_l0_reverse.data)
        init.kaiming_normal_(self.rnn.weight_ih_l0_reverse.data)
        init.constant_(self.rnn.bias_hh_l0_reverse.data, val=0)
        init.constant_(self.rnn.bias_ih_l0_reverse.data, val=0)

        init.orthogonal_(self.rnn.weight_hh_l1.data)
        init.kaiming_normal_(self.rnn.weight_ih_l1.data)
        init.constant_(self.rnn.bias_hh_l1.data, val=0)
        init.constant_(self.rnn.bias_ih_l1.data, val=0)
        init.orthogonal_(self.rnn.weight_hh_l1_reverse.data)
        init.kaiming_normal_(self.rnn.weight_ih_l1_reverse.data)
        init.constant_(self.rnn.bias_hh_l1_reverse.data, val=0)
        init.constant_(self.rnn.bias_ih_l1_reverse.data, val=0)
        if self.rnn_type == 'lstm':
            # Set the initial forget bias values to 1
            self.rnn.bias_ih_l0.data.chunk(4)[1].fill_(1)
            self.rnn.bias_ih_l0_reverse.data.chunk(4)[1].fill_(1)

            self.rnn.bias_ih_l1.data.chunk(4)[1].fill_(1)
            self.rnn.bias_ih_l1_reverse.data.chunk(4)[1].fill_(1)
    def forward(self, inputs, length):
        """
        Args:
            inputs (Variable): A float variable of size
                (max_length, batch_size, input_dim).
            length (Tensor): A long tensor of sequence lengths.

        Returns:
            output (Variable): An encoded sequence vector of size
                (batch_size, hidden_dim).
        """

        inputs_packed = pack_padded_sequence(inputs, lengths=list(length))
        rnn_outputs_packed, _ = self.rnn(inputs_packed)
        rnn_outputs, _ = pad_packed_sequence(rnn_outputs_packed)
        # To avoid the weired bug when taking the max of a length-1 sentence.
        output = rnn_outputs.max(dim=0, keepdim=True)[0].squeeze(0)
        return output
