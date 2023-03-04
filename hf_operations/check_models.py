from pprint import pprint
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='vocabtrimmer')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models]

print("\nFoundation Model")
pprint(sorted([i for i in models_filtered if 'quad' not in i and 'mt5-small' in i], key=lambda x: int(x.split('-')[-1])))
pprint(sorted([i for i in models_filtered if 'quad' not in i and 'mt5-base' in i], key=lambda x: int(x.split('-')[-1])))
pprint(sorted([i for i in models_filtered if 'quad' not in i and 'xlm' in i], key=lambda x: int(x.split('-')[-1])))

print("\nFT QG")
pprint(sorted([i for i in models_filtered if 'quad' in i and 'mt5-small' in i and i.endswith("-qg")], key=lambda x: int(x.split('-')[-3])))

print("\nTrimmed QG")
pprint(sorted([i for i in models_filtered if 'qg-trimmed' in i and 'mt5-small' in i], key=lambda x: int(x.split('-')[-1])))

# pprint(sorted([i for i in models_filtered if 'quad' in i]))
#
# pprint(sorted([i for i in models_filtered if 'quad' not in i]))

# input("delete all? >>>")
# for i in models_filtered:
#     api.delete_repo(repo_id=i, repo_type='model')
