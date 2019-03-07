import torch.nn as nn


class CustomedBiLstm(nn.Module):
    def __init__(self,
                 alphabet_size,
                 vocab_size,
                 word_embedding_dim,
                 char_embedding_dim,
                 n_hidden,
                 n_tags):
        super(CustomedBiLstm, self).__init__()
        self.alphabet_size = alphabet_size
        self.vocab_size = vocab_size
        self.word_embedding_dim = word_embedding_dim
        self.char_embedding_dim = char_embedding_dim
        self.n_hidden = n_hidden
        self.n_tags = n_tags

        self.char_embedding_layer = nn.Embedding(self.alphabet_size, self.char_embedding_dim)
        self.lower_LSTM = nn.LSTM(input_size=self.char_embedding_dim, hidden_size=self.n_hidden,
                                  num_layers=self.n_hidden)
        self.word_embedding_layer = nn.Embedding(self.vocab_size, self.word_embedding_dim)
        self.upper_LSTM = nn.LSTM(input_size=self.n_hidden + self.word_embedding_dim, hidden_size=self.n_hidden,
                                  bidirectional=True)
        self.hidden_to_tag = nn.Linear(self.n_hidden*2, self.n_tags)

    def forward(self, tokens_tensor, char_tensor):
        print('char_tensor', char_tensor.shape)
        char_embeds = self.char_embedding_layer(char_tensor)
        print('char_embeds', char_embeds.shape)
        lower_lstm_out, hidden = self.lower_LSTM(char_embeds)
        print(lower_lstm_out.shape)
        tokens_tensor = self.word_embedding_layer(tokens_tensor)
        print('tokens_tensor', tokens_tensor.shape)
        return
