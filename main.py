import argparse

from src.trainer import trainer
from src.util.data import get_languages
from torch import cuda

languages = [t for t in get_languages()]
languages_arguments = languages + ['all']

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the bi-LSTM POS tagger')
    main_group = parser.add_argument_group('Main', 'Main arguments')
    main_group.add_argument('--language',
                            help='Language dataset to be used among: ' + ' | '.join(languages_arguments),
                            type=str,
                            required=True)
    main_group.add_argument('--gpu',
                            help='Use GPU (default: False)',
                            type=bool,
                            default=False)
    main_group.add_argument('--save_model',
                            help='Save the model(s) (default: False)',
                            type=bool,
                            default=False)
    # TODO: Running shell script from within Python can be unsafe?
    # parser.add_argument('--download_data', help='Download the Universal Dependency Dataset. '
    #                     'This will delete all the cached files in the data folder including built vocab and alphabet '
    #                     '(default: False)',
    #                     type=bool,
    #                     default=False)
    configs = parser.add_argument_group('Config', 'Model config')
    configs.add_argument('--word_embed_dim',
                         help='Word embedding dimension (default: 128)',
                         type=int,
                         default=128)
    configs.add_argument('--char_embed_dim',
                         help='Character embedding dimension (default: 100)',
                         type=int,
                         default=100)
    configs.add_argument('--char_hidden_dim',
                         help='Character level LSTM hidden dimension (default: 100)',
                         type=int,
                         default=100)
    configs.add_argument('--word_hidden_dim',
                         help='Word level LSTM hidden dimension (default: 100)',
                         type=int,
                         default=100)

    configs = parser.add_argument_group('Trainer', 'Train config')
    configs.add_argument('--n_epochs',
                         help='Number of training epochs (default: 20)',
                         type=int,
                         default=20)
    configs.add_argument('--optimizer',
                         help='Optimzer choice among: Adam | SGD (default: Adam)',
                         type=str,
                         default='Adam')
    configs.add_argument('--lr',
                         help='Learning rate (default: 0.0001)',
                         type=float,
                         default=0.0001)

    try:
        args = parser.parse_args()
    except:
        parser.print_help()
        exit()

    use_gpu = args.gpu and cuda.is_available()

    configs = {'n_epochs': args.n_epochs, 'word_embed_dim': args.word_embed_dim, 'char_embed_dim': args.char_embed_dim,
               'char_hidden_dim': args.char_hidden_dim, 'word_hidden_dim': args.word_hidden_dim,
               'optimizer': args.optimizer, 'lr': args.lr, 'use_gpu': use_gpu, 'save_model': args.save_model}

    if args.language == 'all':
        for lang in languages:
            trainer(lang, configs)
    else:
        trainer(args.language, configs)



