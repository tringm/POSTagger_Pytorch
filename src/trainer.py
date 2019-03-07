from random import shuffle

import timeit

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from src.model import CustomedBiLstm
from src.util.data import LanguageDataset, SplitData
from src.util.data import get_languagues
from src.util.timer import f_timer


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
        tokens_as_char[idx] = tok
    char_tensor = torch.stack([c_tnsr for c_tnsr in tokens_as_char], dim=1)
    tags_tensor = torch.LongTensor([all_tags.index(tag) for tag in tags])
    return tokens_tensor, char_tensor, tags_tensor


def evaluate(split_dataset, model, vocab, alphabet, all_tags):
    correct = 0
    preds = []
    actuals = []

    for idx in range(len(split_dataset.tokens)):
        tokens = split_dataset.tokens[idx]
        tags = split_dataset.tags[idx]
        tokens_tensor, char_tensor, tags_tensor = get_one_batch(tokens, tags, vocab, alphabet, all_tags)
        log_probs = model(tokens_tensor, char_tensor)
        _, predicted = torch.max(log_probs, dim=1)
        preds.append(predicted.data.numpy().tolist())
        actuals.append(tags_tensor.data.numpy().tolist())

    actuals = MultiLabelBinarizer(classes=np.arange(len(all_tags))).fit_transform(actuals)
    preds = MultiLabelBinarizer(classes=np.arange(len(all_tags))).fit_transform(preds)

    return accuracy_score(actuals, preds)



def trainer(language, model_choice, config=None):
    if not config:
        config = {'n_epochs': 20, 'word_embedding_dim': 128, 'char_embedding_dim': 100, 'n_hidden': 100,
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

    train_split: SplitData = lang_data.train_split

    for epoch in range(config['n_epochs']):
        start_epoch = timeit.default_timer()
        indices = np.arange(len(train_split.tokens))
        shuffle(indices)
        train_tokens = [train_split.tokens[idx] for idx in indices]
        train_tags = [train_split.tags[idx] for idx in indices]

        total_loss = 0
        model.zero_grad()

        for idx in range(len(train_split.tokens)):
            tokens_tensor, char_tensor, tags_tensor = get_one_batch(train_tokens[idx], train_tags[idx],
                                                                    vocab, alphabet, all_tags)
            log_probs = model(tokens_tensor, char_tensor)
            batch_loss = loss_function(log_probs, tags_tensor)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss

        print('epoch: %d, loss: %.4f' % ((epoch + 1), total_loss))

        start_evaluate_train = timeit.default_timer()
        train_acc = evaluate(lang_data.train_split, model, vocab, alphabet, all_tags)
        print('evaluation train took %.4f' % (timeit.default_timer() - start_evaluate_train))
        start_evaluate_dev = timeit.default_timer()
        dev_acc = evaluate(lang_data.dev_split, model, vocab, alphabet, all_tags)
        print('evaluation dev took %.4f' % (timeit.default_timer() - start_evaluate_dev))

        print('epoch: %d, loss: %.4f, train acc: %.2f%%, dev acc: %.2f%%' %
              (epoch + 1, total_loss, train_acc, dev_acc))

        print('One epoch took %.4f' %(timeit.default_timer() - start_epoch))


trainer('Vietnamese', 'bilstm')
