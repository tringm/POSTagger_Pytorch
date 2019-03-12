import torch.nn as nn
import torch
import torch.nn.functional as F


class CustomedBiLstm(nn.Module):
    def __init__(self,
                 alphabet_size,
                 vocab_size,
                 word_embed_dim,
                 char_embed_dim,
                 char_hidden_dim,
                 word_hidden_dim,
                 n_tags):
        super(CustomedBiLstm, self).__init__()
        self.alphabet_size = alphabet_size
        self.vocab_size = vocab_size
        self.word_embed_dim = word_embed_dim
        self.char_embed_dim = char_embed_dim
        self.char_hidden_dim = char_hidden_dim
        self.word_hidden_dim = word_hidden_dim
        self.n_tags = n_tags

        self.char_embedding_layer = nn.Embedding(self.alphabet_size, self.char_embed_dim)
        self.lower_LSTM = nn.LSTM(input_size=self.char_embed_dim,
                                  hidden_size=self.char_hidden_dim,
                                  batch_first=True)
        self.word_embedding_layer = nn.Embedding(self.vocab_size, self.word_embed_dim)
        self.upper_LSTM = nn.LSTM(input_size=self.char_hidden_dim + self.word_embed_dim,
                                  hidden_size=self.word_hidden_dim,
                                  bidirectional=True)
        self.hidden_to_tag = nn.Linear(self.word_hidden_dim*2, self.n_tags)

    def forward(self, tokens_tensor, char_tensor):
        char_embeds = self.char_embedding_layer(char_tensor)
        lower_lstm_out, hidden = self.lower_LSTM(char_embeds)
        last_state_lower_lstm = lower_lstm_out[-1]
        tokens_embeds = self.word_embedding_layer(tokens_tensor)

        final_embeds = torch.cat((last_state_lower_lstm, tokens_embeds), 1).view(tokens_tensor.shape[0], 1, -1)
        upper_lstm_out, hidden = self.upper_LSTM(final_embeds)
        out = self.hidden_to_tag(upper_lstm_out.view(tokens_tensor.shape[0], -1))
        out = F.log_softmax(out, dim=1)
        return out
