### Description
This project aims to reimplement the [Multilingual Part-of-Speech Tagging Model with
Bidirectional Long Short-Term Memory Models and Auxiliary Loss](https://arxiv.org/pdf/1604.05529.pdf) using pytorch.

### The dataset
Multi-lingual tree banks, over 70 languages.
[Link](https://universaldependencies.org/#download)

The dataset can be download by running the ```.downloadData.sh``` bash script

### Project Structure
#### Util
Contains different helper for the project
1. Data: Contains different helper for loading and processing the dataset:
    * ```get_languages```: Since a language can have multiple repo, I chosed the language with both train, dev, and test set with largest number of tokens (based on its correspondint 'stats.xml'). The languages and its repo will be cached in ```lang_to_dir.pkl``` in data directory.
    * ```LanguageDataset```: a customed language loader class to load a language. This class contains:
        * 3 splits of a dataset as SplitData Object
        * The dataset vocab, cached in a pkl file
        * The dataset alphabet, , cached in a pkl file
    * ```SplitData``` contains data of a specific split, cached in a pkl file:
        * conllu: conllu data loaded using the [conllu parser lib](https://github.com/EmilStenstrom/conllu) 
        * tokens: list of tokens for each sentences
        * tags: list of corresponding tags
        * vocab: vocab built using torcthext
        * alphabet: alphabet using torchtext
1. NLP: NLP Helper library
    * ```build_vocab_from_sentences_tokens```: used torchtext to build vocab from list of conllu loaded tokens
    * ```build_alphabet_from_sentences_tokens```: similarly, used torchtext to build alphabet from list of conllu loaded tokens 

#### The model
. The Model Bi-directional LSTM
