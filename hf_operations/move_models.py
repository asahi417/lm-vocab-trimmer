from pprint import pprint
import os
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='vocabtrimmer')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if 'mt5-small' in i.modelId and "qg-trimmed" in i.modelId]

la = [i.split("quad-")[0][-2:] for i in models_filtered]
new = [f"trimmed-{l}".join(i.split("trimmed")) for i, l in zip(models_filtered, la)]
pprint(list(zip(models_filtered, new)))
for i, n in zip(models_filtered, new):
    api.move_repo(from_id=i, to_id=n, repo_type='model')
