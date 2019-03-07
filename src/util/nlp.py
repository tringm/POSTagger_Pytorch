from torchtext import data


def build_vocab_from_sentences_tokens(sentences_tokens):
    """
    use torch text to build vocab object from a list of sentences that is already tokenized in to tokens
    :param sentences_tokens: list of list of tokens
    :return: torchtext.vocab object
    """
    token_field = data.Field(tokenize=list, init_token='<root>')
    fields = [('tokens', token_field)]
    examples = [data.Example.fromlist([t], fields) for t in sentences_tokens]
    torch_dataset = data.Dataset(examples, fields)
    token_field.build_vocab(torch_dataset)
    return token_field.vocab


def build_alphabet_from_sentence_tokens(sentences_tokens):
    """
    Build alphabet from tokens by converting tokens to character
    :param sentences_tokens:
    :return:
    """
    def to_char(tokens):
        return [c for tok in tokens for c in list(tok)]
    sentences_char = [to_char(sent) for sent in sentences_tokens]
    char_field = data.Field(tokenize=list, init_token='<root>')
    fields = [('char', char_field)]
    examples = [data.Example.fromlist([t], fields) for t in sentences_char]
    torch_dataset = data.Dataset(examples, fields)
    char_field.build_vocab(torch_dataset)
    return char_field.vocab
