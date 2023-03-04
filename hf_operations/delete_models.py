from pprint import pprint
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='vocabtrimmer')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models]

pprint(sorted([i for i in models_filtered if 'quad' not in i]))

input("delete all? >>>")
for i in models_filtered:
    api.delete_repo(repo_id=i, repo_type='model')
