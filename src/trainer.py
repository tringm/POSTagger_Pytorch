import torch.nn as nn
import torch.optim as optim
import torch

from src.model import CustomedBiLstm
from src.util.data import LanguageDataset, SplitData
from src.util.data import get_languagues
from src.util.timer import f_timer
import numpy as np

from random import shuffle


def get_one_batch(tokens, tags, vocab, alphabet, all_tags):
    def token_to_char_tensor(tok):
        return torch.LongTensor([alphabet.stoi[c] for c in tok])

    tokens_tensor = torch.LongTensor([vocab.stoi[tok] for tok in tokens])

    # Perform padding for tokens
    max_tok_len = max([len(tok) for tok in tokens])
    tokens_as_char = [token_to_char_tensor(tok) for tok in tokens]
    for idx, tok in enumerate(tokens_as_char):
        if len(tok) < max_tok_len:
            paddings = torch.from_numpy(np.full(max_tok_len - len(tok), alphabet.stoi['<pad>']))
            tok = torch.cat((tok, paddings))
        tokens_as_char[idx] = tok.view(-1)
    char_tensor = torch.stack([c_tnsr for c_tnsr in tokens_as_char], dim=0)
    tags_tensor = torch.Tensor([all_tags.index(tag) for tag in tags])
    return tokens_tensor, char_tensor, tags_tensor


def trainer(language, model_choice, config=None):
    if not config:
        config = {'n_epochs': 1, 'word_embedding_dim': 128, 'char_embedding_dim': 100, 'n_hidden': 100,
                  'optimizer_choice': 'SGD', 'lr': 0.1}

    all_languages = f_timer(print, get_languagues)
    if language not in all_languages:
        raise ValueError(f'language {language} not found')
    lang_data: LanguageDataset = all_languages[language]

    vocab = lang_data.vocab
    alphabet = lang_data.alphabet

    meta = lang_data.meta
    all_tags = meta['all_tags']
    vocab_size = meta['n_tokens']
    n_tags = meta['n_tags']
    model = CustomedBiLstm(alphabet_size=len(alphabet), vocab_size=len(vocab), word_embedding_dim=config['word_embedding_dim'],
                           char_embedding_dim=config['char_embedding_dim'], n_hidden=config['n_hidden'], n_tags=n_tags)

    loss_function = nn.NLLLoss()
    if config['optimizer_choice'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer_choice'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])

    dev_split: SplitData = lang_data.dev_split

    for epoch in range(config['n_epochs']):
        indices = np.arange(len(dev_split.tokens))
        shuffle(indices)
        dev_tokens = [dev_split.tokens[idx] for idx in indices]
        dev_tags = [dev_split.tags[idx] for idx in indices]

        total_loss = 0
        model.zero_grad()

        for idx in range(len(dev_split.tokens)):
            tokens = dev_tokens[idx]
            tags = dev_tags[idx]

            print('tokens', tokens)

            tokens_tensor, char_tensor, tags_tensor = get_one_batch(tokens, tags, vocab, alphabet, all_tags)
            log_probs = model(tokens_tensor, char_tensor)
            batch_loss = loss_function(log_probs, tags_tensor)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss


trainer('Vietnamese', 'bilstm')
