import torch.nn as nn
import torch.nn.functional as F

class POSTagger(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_dim,
                 hidden_dim,
                 target,
                 bidirectional=False
                 ):
        super(POSTagger, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.target = target

        self.embedding_layer = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim,
                            bidirectional=bidirectional)
        self.hidden_to_tag = nn.Linear(self.hidden_dim, self.target)

    def forward(self, sentences, range, length):
        out = self.embedding_layer(sentences)
        out, hidden = self.lstm(out)




