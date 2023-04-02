from pprint import pprint
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='vocabtrimmer')
# filt = ModelFilter(author='cardiffnlp')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models]

# pprint(sorted([i for i in models_filtered if "xlm-roberta-base" in i]))
#
# pprint(sorted([i for i in models_filtered if "mt5" in i]))
#
# pprint(sorted([i for i in models_filtered if "mbart" in i]))

pprint(sorted([i for i in models_filtered if "xlm-roberta" in i]))


