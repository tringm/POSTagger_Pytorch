import torch.nn as nn

class CustomedBiLstm(nn.Module):
    def __init__(self,
                 word_max_length,
                 vocab_size,
                 word_embedding_dim,
                 char_embedding_dim,
                 n_hidden,
                 n_tags):
        super(CustomedBiLstm, self).__init__()
        self.word_max_length = word_max_length
        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.n_hidden = n_hidden
        self.n_tags = n_tags

        self.char_embedding_layer = nn.Embedding(self.word_max_length, self.char_embedding_dim)
        self.lower_LSTM = nn.LSTM(input_size=self.char_embedding_dim, hidden_size=self.n_hidden,
                                  num_layers=self.n_hidden)
        self.word_embedding_layer = nn.Embedding(self.vocab_size, self.word_embedding_dim)
        self.upper_LSTM = nn.LSTM(input_size=self.n_hidden + self.word_embedding_dim, hidden_size=self.n_hidden,
                                  bidirectional=True)
        self.hidden_to_tag = nn.Linear(self.n_hidden*2, self.n_tags)

    def forward(self, sentence):
        return
