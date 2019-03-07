from src.util.data import LanguageDataset, SplitData
from src.model import CustomedBiLstm
from src.util.data import get_languagues, sentence_to_tokens_and_tags
from src.util.nlp import word_to_unicodes
from src.util.timer import f_timer

import torch.nn as nn
import torch.optim as optim


def trainer(language, model_choice, config=None):
    if not config:
        config = {'n_epochs': 20, 'word_embedding_dim': 128, 'char_embedding_dim': 100, 'n_hidden': 100,
                  'optimizer_choice': 'SGD', 'lr': 0.1}

    all_languages = f_timer(print, get_languagues)
    if language not in all_languages:
        raise ValueError(f'language {language} not found')
    lang_data: LanguageDataset = all_languages[language]

    meta = lang_data.meta

    all_tags = meta['all_tags']
    vocab_size = meta['n_tokens']
    n_tags = meta['n_tags']

    model = CustomedBiLstm(word_max_length=10, vocab_size=vocab_size, word_embedding_dim=config['word_embedding_dim'],
                           char_embedding_dim=config['char_embedding_dim'], n_hidden=config['n_hidden'], n_tags=n_tags)

    loss_function = nn.NLLLoss()
    if config['optimizer_choice'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    elif config['optimizer_choice'] == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=config['lr'])

    dev_split: SplitData = lang_data.dev_split


    sentence = dev_split.conllu_contents[0]
    tokens, tags = sentence_to_tokens_and_tags(sentence)
    print(tokens)
    print(tags)



    # for epoch in range(config['n_epochs']):
    #     train(model, )

trainer('Vietnamese', 'bilstm')
