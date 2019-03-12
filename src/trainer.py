import json
import timeit
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

from config import root_path
from src.model import CustomedBiLstm
from src.util.data import LanguageDataset, SplitData
from src.util.data import get_languages
from src.util.misc import f_timer


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
    accuracy = []

    for idx in range(len(split_dataset.tokens)):
        tokens = split_dataset.tokens[idx]
        tags = split_dataset.tags[idx]
        tokens_tensor, char_tensor, tags_tensor = get_one_batch(tokens, tags, vocab, alphabet, all_tags)
        log_probs = model(tokens_tensor, char_tensor)
        _, predicted = torch.max(log_probs, dim=1)
        accuracy.append(accuracy_score(tags_tensor.data.numpy().tolist(), predicted.data.numpy().tolist()))
    return sum(accuracy) / len(accuracy)


def trainer(language, configs):
    all_languages, _ = f_timer(get_languages)
    if language not in all_languages:
        raise ValueError(f'language {language} not found')
    lang_data: LanguageDataset = all_languages[language]

    vocab = lang_data.vocab
    alphabet = lang_data.alphabet

    meta = lang_data.meta
    all_tags = meta['all_tags']
    n_tags = meta['n_tags']
    model = CustomedBiLstm(alphabet_size=len(alphabet), vocab_size=len(vocab), word_embed_dim=configs['word_embed_dim'],
                           char_embed_dim=configs['char_embed_dim'], char_hidden_dim=configs['char_hidden_dim'],
                           word_hidden_dim=configs['word_hidden_dim'], n_tags=n_tags)

    loss_function = nn.NLLLoss()
    if configs['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=configs['lr'])
    elif configs['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=configs['lr'])

    train_split: SplitData = lang_data.train_split

    results = {}

    with (root_path() / 'src' / 'out' / 'log' / (lang_data.name + '.log')).open(mode='w') as f:
        results['Language'] = lang_data.name
        results['Config'] = configs
        f.write(f"Language: {lang_data.name} \n")
        f.write(f"Model: {model} \n")
        f.write(f"Config: {configs}")
        results['Model'] = str(model)
        results['Time'] = []
        results['Performance'] = []
        for epoch in range(configs['n_epochs']):
            epoch_time = {'epoch': epoch + 1}
            epoch_perf = {'epoch': epoch + 1}
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

            training_time = timeit.default_timer() - start_epoch
            f.write('\t traing the model with %d sample took %.4f \n' % (len(train_split.tokens), training_time))
            epoch_time['train'] = training_time

            train_acc, train_eval_time = f_timer(evaluate, lang_data.train_split, model, vocab, alphabet, all_tags)
            dev_acc, test_eval_time = f_timer(evaluate, lang_data.dev_split, model, vocab, alphabet, all_tags)
            f.write('\t evaluation train split took %.4f \n' % train_eval_time)
            f.write('\t evaluation dev took %.4f \n' % test_eval_time)
            epoch_time['train_eval'] = train_eval_time
            epoch_time['test_eval'] = test_eval_time

            f.write('\t one epoch took %.4f \n' % (timeit.default_timer() - start_epoch))
            f.write('epoch: %d, loss: %.4f, train acc: %.3f, dev acc: %.3f \n' %
                    (epoch + 1, total_loss, train_acc, dev_acc))
            epoch_perf['loss'] = ("%.4f" % total_loss)
            epoch_perf['train_acc'] = ("%.3f" % train_acc)
            epoch_perf['dev_acc'] = ("%.3f" % dev_acc)

            results['Time'].append(epoch_time)
            results['Performance'].append(epoch_perf)

        test_acc = evaluate(lang_data.test_split, model, vocab, alphabet, all_tags)
        f.write('test acc: %.3f%% \n' % test_acc)
        results['Accuracy'] = test_acc

    with (root_path() / 'src' / 'out' / 'test' / (lang_data.name + '.json')).open(mode='w') as f:
        json.dump(results, f, indent=4, sort_keys=True)
