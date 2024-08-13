from pprint import pprint
import os
from huggingface_hub import HfApi

api = HfApi()
models = api.list_models(author='vocabtrimmer')
models_filtered = [i.id for i in models if 'xlm-roberta-base' in i.id]
models_filtered = [i for i in models_filtered if "sentiment" in i and "trimmed" not in i]

pprint(models_filtered)
input()
for i in models_filtered:
    target = i.replace("vocabtrimmer", "cardiffnlp")
    api.move_repo(from_id=i, to_id=target, repo_type='model')
