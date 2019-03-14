import json
import logging
import timeit
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score

from config import root_path
from src.model import CustomedBiLstm
from src.util.data import LanguageDataset
from src.util.data import get_languages
from src.util.misc import f_timer


def get_one_batch(tokens, tags, vocab, alphabet, all_tags):
    def token_to_char_tensor(tok):
        tensor = torch.LongTensor([alphabet.stoi[c] for c in tok])
        return tensor

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


def evaluate(split_dataset, model, vocab, alphabet, all_tags, use_gpu):
    accuracy = []

    for idx in range(len(split_dataset.tokens)):
        tokens = split_dataset.tokens[idx]
        tags = split_dataset.tags[idx]
        tokens_tensor, char_tensor, tags_tensor = get_one_batch(tokens, tags, vocab, alphabet, all_tags)
        if use_gpu:
            tokens_tensor = tokens_tensor.cuda()
            char_tensor = char_tensor.cuda()
        log_probs = model(tokens_tensor, char_tensor)
        _, predicted = torch.max(log_probs, dim=1)
        if use_gpu:
            predicted = predicted.cpu()
        accuracy.append(accuracy_score(tags_tensor.data.numpy().tolist(), predicted.data.numpy().tolist()))
    return sum(accuracy) / len(accuracy)


def trainer(lang_data, configs):
    vocab = lang_data.vocab
    alphabet = lang_data.alphabet

    meta = lang_data.meta
    use_gpu = configs['use_gpu']
    model = CustomedBiLstm(alphabet_size=len(alphabet), vocab_size=len(vocab), word_embed_dim=configs['word_embed_dim'],
                           char_embed_dim=configs['char_embed_dim'], char_hidden_dim=configs['char_hidden_dim'],
                           word_hidden_dim=configs['word_hidden_dim'], n_tags=meta['n_tags'], use_gpu=use_gpu)
    if use_gpu:
        model.cuda()

    loss_function = nn.NLLLoss()
    if configs['optimizer'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=configs['lr'])
    elif configs['optimizer'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=configs['lr'])

    n_try = 0
    log_path = root_path() / 'src' / 'out' / 'log' / (lang_data.name + '_' + str(n_try) + '.log')
    while log_path.exists():
        n_try += 1
        log_path = root_path() / 'src' / 'out' / 'log' / (lang_data.name + '_' + str(n_try) + '.log')
    logging.basicConfig(filename=str(log_path), level=logging.INFO)
    logging.getLogger('trainer')

    results = {'Language':lang_data.name,
               'Repo': lang_data.repo.stem,
               'Stats': {'n_tokens': meta['n_tokens'],
                         'n_train': len(lang_data.train_split.tokens),
                         'n_dev': len(lang_data.dev_split.tokens),
                         'n_test': len(lang_data.test_split.tokens)},
               'Config': configs,
               'Model': str(model),
               'Time': [],
               'Performance': []
               }
    logging.info(f"Language: {lang_data.name} \n")
    logging.info(f"Repo: {lang_data.repo} \n")
    logging.info(f"Number of tokens: {meta['n_tokens']} \n")
    logging.info(f"Train size: {len(lang_data.train_split.tokens)}, "
                 f"Dev size: {len(lang_data.dev_split.tokens)}, "
                 f"Test size: {len(lang_data.test_split.tokens)}")
    logging.info(f"Model: {model} \n")
    logging.info(f"Config: {configs}\n")

    for epoch in range(configs['n_epochs']):
        logging.info(f"epoch: {epoch}\n")
        epoch_time = {'epoch': epoch + 1}
        epoch_perf = {'epoch': epoch + 1}
        start_epoch = timeit.default_timer()

        indices = np.arange(len(lang_data.train_split.tokens))
        shuffle(indices)
        train_tokens = [lang_data.train_split.tokens[idx] for idx in indices]
        train_tags = [lang_data.train_split.tags[idx] for idx in indices]

        total_loss = 0
        model.zero_grad()
        for idx in range(len(lang_data.train_split.tokens)):
            tokens_tensor, char_tensor, tags_tensor = get_one_batch(train_tokens[idx], train_tags[idx],
                                                                    vocab, alphabet, meta['all_tags'])
            if use_gpu:
                tokens_tensor = tokens_tensor.cuda()
                char_tensor = char_tensor.cuda()
                tags_tensor = tags_tensor.cuda()
            log_probs = model(tokens_tensor, char_tensor)
            batch_loss = loss_function(log_probs, tags_tensor)
            batch_loss.backward()
            optimizer.step()
            total_loss += batch_loss

        training_time = timeit.default_timer() - start_epoch
        logging.info('\t training the model took %.4f \n' % training_time)
        epoch_time['train'] = training_time

        train_acc, train_eval_time = f_timer(evaluate, lang_data.train_split, model, vocab, alphabet,
                                             meta['all_tags'], use_gpu)
        dev_acc, test_eval_time = f_timer(evaluate, lang_data.dev_split, model, vocab, alphabet,
                                          meta['all_tags'], use_gpu)
        logging.info('\t evaluation train split took %.4f \n' % train_eval_time)
        logging.info('\t evaluation dev took %.4f \n' % test_eval_time)
        epoch_time['train_eval'] = train_eval_time
        epoch_time['test_eval'] = test_eval_time

        logging.info('\t one epoch took %.4f \n' % (timeit.default_timer() - start_epoch))
        logging.info('\t loss: %.4f, train acc: %.3f, dev acc: %.3f \n' %
                (total_loss, train_acc, dev_acc))
        epoch_perf['loss'] = ("%.4f" % total_loss)
        epoch_perf['train_acc'] = ("%.3f" % train_acc)
        epoch_perf['dev_acc'] = ("%.3f" % dev_acc)

        results['Time'].append(epoch_time)
        results['Performance'].append(epoch_perf)

    test_acc = evaluate(lang_data.test_split, model, vocab, alphabet, meta['all_tags'], use_gpu)
    logging.info('test acc: %.3f%% \n' % test_acc)
    results['Accuracy'] = test_acc

    with (root_path() / 'src' / 'out' / 'test' / (lang_data.name + '.json')).open(mode='w') as f:
        json.dump(results, f, indent=4, sort_keys=True)

    if configs['save_model']:
        model_name = lang_data.name + '.model'
        torch.save(model, root_path() / 'src' / 'out' / 'cache' / model_name)
