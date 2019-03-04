from itertools import groupby
from operator import itemgetter
from config import root_path
import xmltodict
import numpy as np
import conllu
import pickle
from pathlib import Path

data_path = root_path() / 'data' / 'ud-treebanks-v2.3'


def get_languagues():
    """
    Find languages in the dataset that has train, dev, and test set.
    If the languages has multiple dataset, chose the directory with the largest amount of tokens
    :return: array of tuples of lang and dir
    """
    lang_to_dir_path = root_path() / 'data' / 'lang_to_dir.pkl'
    if lang_to_dir_path.exists():
        with lang_to_dir_path.open(mode='rb') as f:
            return pickle.load(f)

    # find datasets with train, dev, and test split
    all_dir = [(dir.name.split('-')[0][3:], dir) for dir in data_path.iterdir()
               if len(list(dir.glob('*.conllu'))) > 2]
    # groupby language name
    lang_dir = [(lang, list(list(zip(*dirs))[1])) for lang, dirs in groupby(all_dir, itemgetter(0))]
    # get directory with the most amount of tokens
    for idx, t in enumerate(lang_dir):
        lang = t[0]
        list_dirs = t[1]
        if len(list_dirs) == 1:
            lang_dir[idx] = (lang, list_dirs[0])
        else:
            lang_stats = []
            for dir in list_dirs:
                with (dir / 'stats.xml').open() as f:
                    stats = xmltodict.parse(f.read())
                    lang_stats.append(stats['treebank']['size']['total']['tokens'])
            lang_dir[idx] = (lang, list_dirs[np.argmax(lang_stats)])
    lang_dir = dict(lang_dir)
    with lang_to_dir_path.open(mode='wb') as f:
        pickle.dump(lang_dir, f)
    return lang_dir


def load_conllu_file(path):
    # TODO: try conllu.parse_incr for not loading the whole file into memory
    with path.open(mode='r') as f:
        sentences = conllu.parse(f.read())
    return sentences


def load_language(lang):
    """
    Load a language dataset with the largest amount of tokens
    :param lang:
    :return:
    """
    dir = get_languagues()[lang]
    files = list(dir.glob('*.conllu'))
    print(files)
    dev_sentences = load_conllu_file([f for f in files if 'dev' in f.name][0])
    train_sentences = load_conllu_file([f for f in files if 'train' in f.name][0])
    test_sentences = load_conllu_file([f for f in files if 'test' in f.name][0])

    # Load meta

    return dev_sentences, train_sentences, test_sentences

# print(get_languagues())
# load_language('Japanese')
