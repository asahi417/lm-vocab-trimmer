import os
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='relbert')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if 'feature-extraction' in i.tags and i.modelId.startswith('relbert') and i.modelId != 'relbert/relbert-roberta-large']
target = 'research-backup'

for i in models_filtered:
    api.move_repo(from_id=i, to_id=f"{target}/{os.path.basename(i)}", repo_type='model')
