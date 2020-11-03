from torch import nn
from torch.nn import init

from .birnn_max import BiRNNMax


class BiRNNTextClassifier(nn.Module):

    def __init__(self, rnn_type, num_classes, word_dim, hidden_dim,
                 clf_dim, dropout_prob=0):
        super().__init__()
        self.rnn_type = rnn_type
        self.num_classes = num_classes
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.clf_dim = clf_dim
        self.dropout_prob = dropout_prob

        self.dropout = nn.Dropout(dropout_prob)

        self.birnn_max = BiRNNMax(
            rnn_type=rnn_type, input_dim=word_dim, hidden_dim=hidden_dim,
            dropout_prob=dropout_prob)
        self.clf = nn.Sequential(
            nn.Linear(in_features=2 * hidden_dim, out_features=clf_dim),
            nn.ReLU(),
            self.dropout,
            nn.Linear(in_features=clf_dim, out_features=num_classes))
        self.reset_parameters()

    def reset_parameters(self):
        self.birnn_max.reset_parameters()
        init.kaiming_normal_(self.clf[0].weight.data)
        init.constant_(self.clf[0].bias.data, val=0)
        init.uniform_(self.clf[3].weight.data, -0.005, 0.005)
        init.constant_(self.clf[3].bias.data, val=0)

    def forward(self, inputs, length, batch_first=False):
        """
        Args:
            inputs (Variable):
                If use_pretrained_embeddings is False, this is a long
                    variable of size (max_length, batch_size) or
                    (batch_size, max_length) (if batch_first) which
                    contains indices of words.
                If use_pretrained_embeddings is True, this is a 3D
                    variable of size (max_length, batch_size, word_dim)
                    or (batch_size, max_length, word_dim).
            length (Tensor): A long tensor of lengths.
            batch_first (bool): If True, sequences in a batch are
                aligned along the first dimension of inputs.

        Returns:
            logit (Variable): A variable containing unnormalized log
                probability for each class.
        """

        if batch_first:
            inputs = inputs.transpose(0, 1)
        inputs = self.dropout(inputs)
        sentence_vector = self.birnn_max(inputs=inputs, length=length)
        sentence_vector = self.dropout(sentence_vector)
        logit = self.clf(sentence_vector)
        return logit
