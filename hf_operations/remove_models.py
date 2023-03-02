from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='asahi417')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if 'trim' in i.modelId]

print(models_filtered)
input("delete all? >>>")

for i in models_filtered:
    api.delete_repo(repo_id=i, repo_type='model')