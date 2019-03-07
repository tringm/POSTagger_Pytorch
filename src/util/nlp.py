from torchtext import data


def build_vocab_from_sentences_tokens(sentences_tokens):
    """
    use torch text to build vocab object from a list of senetences that is already tokenized in to tokens
    :param sentences_tokens: list of list of tokens
    :return: torchtext.vocab object
    """
    token_field = data.Field(tokenize=list, init_token='<root>')
    fields = [('tokens', token_field)]
    examples = [data.Example.fromlist([t], fields) for t in sentences_tokens]
    torch_dataset = data.Dataset(examples, fields)
    token_field.build_vocab(torch_dataset)
    return token_field.vocab


def word_to_unicodes(word):
    return [c for c in word]
