from src.util.data import get_languagues
from src.trainer import trainer

# TODO: More argument config?
all_languages = get_languagues()
for lang in all_languages:
    trainer(lang)

