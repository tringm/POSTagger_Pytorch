import pickle

import conllu
import numpy as np
import xmltodict

from config import root_path
from src.util.nlp import build_vocab_from_sentences_tokens, build_alphabet_from_sentence_tokens

data_path = root_path() / 'data' / 'ud-treebanks-v2.3'


def get_languages():
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
    languages = dict.fromkeys(list(set(t[0] for t in all_dir)))
    for t in all_dir:
        lang = t[0]
        dir = t[1]
        if not languages[lang]:
            languages[lang] = []
        languages[lang].append(dir)

    # get directory with the most amount of tokens
    for lang in languages:
        list_dirs = languages[lang]
        if len(list_dirs) == 1:
            languages[lang] = LanguageDataset(lang, list_dirs[0])
        else:
            lang_stats = []
            for dir in list_dirs:
                with (dir / 'stats.xml').open() as f:
                    stats = xmltodict.parse(f.read())
                    lang_stats.append(stats['treebank']['size']['total']['tokens'])
            languages[lang] = LanguageDataset(lang, list_dirs[np.argmax(lang_stats)])

    with lang_to_dir_path.open(mode='wb') as f:
        pickle.dump(languages, f)
    return languages


def sentence_to_tokens_and_tags(sentence):
    tokens = []
    tags = []
    for idx, token in enumerate(sentence):
        # Ugly fix: This is for cases when there is missing word or missing tags.
        # E.g:1-2	No	_	_	_	_	_	_	_	_ (Galician-CTG-dev first line)
        if token['form'] == '_' or token['upostag'] == '_':
            continue
        tags.append(token['upostag'])
        tokens.append(token['form'])
    return tokens, tags


def load_conllu_file(path):
    # TODO: try conllu.parse_incr for not loading the whole file into memory
    with path.open(mode='r') as fp:
        sentences = conllu.parse(fp.read())
    return sentences


class LanguageDataset:
    def __init__(self, name, repo):
        self.name = name
        self.repo = repo
        self._splits_data = None
        self._meta = None

    def load_data(self):
        """
        load a dataset based on its repo to dev, train, and test set
        each split will be saved in a pkl file that contains:
        - conllu_contents which is a list of sentences loaded by conllu lib
        - tokens_and_tags which is a list of list of tuples of tokens and tags transformed from sentences
        :return:
        """
        splits_data = dict.fromkeys(['dev', 'train', 'test'])
        raw_files = list(self.repo.glob('*.conllu'))

        for split in splits_data:
            pickle_file = split + '.pkl'
            if (self.repo / pickle_file).exists():
                with (self.repo / pickle_file).open(mode='rb') as f:
                    splits_data[split] = pickle.load(f)
            else:
                try:
                    split_raw_file = [f for f in raw_files if split in f.name][0]
                except IndexError:
                    raise ValueError(f"Raw file for {split} not exist")

                splits_data[split] = SplitData(self, split, split_raw_file)
                with (self.repo / pickle_file).open(mode='wb') as f:
                    pickle.dump(splits_data[split], f)
        return splits_data

    def load_meta(self):
        """
        Load the mete data of the LanguageDataset based on the stats.xml
        The stats includes number of sentence, number of token, number of tag, and all the tags
        :return:
        """
        if (self.repo / 'meta.pkl').exists():
            with (self.repo / 'meta.pkl').open(mode='rb') as f:
                return pickle.load(f)
        else:
            with (self.repo / 'stats.xml').open() as f:
                meta_xml = xmltodict.parse(f.read())['treebank']
            meta = {'n_sentences': int(meta_xml['size']['total']['sentences']),
                    'n_tokens': int(meta_xml['size']['total']['tokens']),
                    'n_tags': int(meta_xml['tags']['@unique']),
                    'all_tags': [t['@name'] for t in meta_xml['tags']['tag']]}

            with (self.repo / 'meta.pkl').open(mode='wb') as f:
                pickle.dump(meta, f)
            return meta

    @property
    def splits_data(self):
        if not self._splits_data:
            self._splits_data = self.load_data()
        return self._splits_data

    @property
    def meta(self):
        if not self._meta:
            self._meta = self.load_meta()
        return self._meta

    @property
    def dev_split(self):
        return self.splits_data['dev']

    @property
    def train_split(self):
        return self.splits_data['train']

    @property
    def test_split(self):
        return self.splits_data['test']

    @property
    def vocab(self):
        if (self.repo / 'vocab.pkl').exists():
            with (self.repo / 'vocab.pkl').open(mode='rb') as f:
                return pickle.load(f)
        else:
            vocab = build_vocab_from_sentences_tokens(self.train_split.tokens)
            with (self.repo / 'vocab.pkl').open(mode='wb') as f:
                pickle.dump(vocab, f)
            return vocab

    @property
    def alphabet(self):
        if (self.repo / 'alphabet.pkl').exists():
            with (self.repo / 'alphabet.pkl').open(mode='rb') as f:
                return pickle.load(f)
        else:
            alphabet = build_alphabet_from_sentence_tokens(self.train_split.tokens)
            with (self.repo / 'alphabet.pkl').open(mode='wb') as f:
                pickle.dump(alphabet, f)
            return alphabet


class SplitData:
    def __init__(self, dataset, name, file):
        self.dataset = dataset
        self.name = name
        self.file = file
        self.conllu_contents = load_conllu_file(self.file)
        if self.conllu_contents[0].tokens[0] == '_':
            raise ValueError(f"Data {self.dataset} is empty, requires merging")
        tokens_and_tags = [sentence_to_tokens_and_tags(sentence) for sentence in self.conllu_contents]
        self.tokens = [t[0] for t in tokens_and_tags]
        self.tags = [t[1] for t in tokens_and_tags]
