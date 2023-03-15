from pprint import pprint
import os
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='vocabtrimmer')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if 'xlm-roberta-base' in i.modelId]
models_filtered = [i for i in models_filtered if "sentiment" in i and "trimmed" not in i]

pprint(models_filtered)
input()
for i in models_filtered:
    target = i.replace("vocabtrimmer", "cardiffnlp")
    api.move_repo(from_id=i, to_id=target, repo_type='model')
