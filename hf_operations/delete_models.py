from pprint import pprint
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='vocabtrimmer')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models]
target = [i for i in models_filtered if "xlm" in i and "tweet" not in i]
pprint(sorted(target))
if len(target) > 0:
    input("delete all? >>>")
    for i in target:
        api.delete_repo(repo_id=i, repo_type='model')


# filt = ModelFilter(author='vocabtrimmer')
# models = api.list_models(filter=filt)
# models_filtered = [i.modelId for i in models]
# target = [i for i in models_filtered if "mt5-small-trimmed-de" in i]
# pprint(sorted(target))
# if len(target) > 0:
#     input("delete all? >>>")
#     for i in target:
#         api.delete_repo(repo_id=i, repo_type='model')
#
# # remove blank models
# target = [i.modelId for i in api.list_models(filter=ModelFilter(author='asahi417'))]
# target = [i for i in target if "trimmed" in i]
# pprint(sorted(target))
# if len(target) > 0:
#     input("delete all? >>>")
#     for i in target:
#         api.delete_repo(repo_id=i, repo_type='model')
#
# # remove blank models
# target = [i.modelId for i in api.list_models(filter=ModelFilter(author='vocabtrimmer')) if 'text2text-generation' not in i.tags and 'xlm' not in i.modelId]
# pprint(sorted(target))
# if len(target) > 0:
#     input("delete all? >>>")
#     for i in target:
#         api.delete_repo(repo_id=i, repo_type='model')
#
# # remove other junk models
# filt = ModelFilter(author='vocabtrimmer')
# models = api.list_models(filter=filt)
# models_filtered = [i.modelId for i in models if 'mt5' in i.modelId]
#
# target = [i for i in models_filtered if '210000' in i or '150000' in i or '180000' in i]
# pprint(sorted(target))
# if len(target) > 0:
#     input("delete all? >>>")
#     for i in target:
#         api.delete_repo(repo_id=i, repo_type='model')
#
# target = [i for i in models_filtered if 'ko' in i and ('120000' in i or '75000' in i or '90000' in i)]
# pprint(sorted(target))
# if len(target) > 0:
#     input("delete all? >>>")
#     for i in target:
#         api.delete_repo(repo_id=i, repo_type='model')
#
# target = [i for i in models_filtered if 'it' in i and ('120000' in i)]
# pprint(sorted(target))
# if len(target) > 0:
#     input("delete all? >>>")
#     for i in target:
#         api.delete_repo(repo_id=i, repo_type='model')
#
# # xlm cleaner
# filt = ModelFilter(author='vocabtrimmer')
# models = api.list_models(filter=filt)
# models_filtered = [i.modelId for i in models if 'xlm' in i.modelId]
# target = [i for i in models_filtered if 'ja' in i]
# pprint(sorted(target))
# if len(target) > 0:
#     input("delete all? >>>")
#     for i in target:
#         api.delete_repo(repo_id=i, repo_type='model')
#
#
# # misc cleaner
# filt = ModelFilter(author='vocabtrimmer')
# models = api.list_models(filter=filt)
# models_filtered = [i.modelId for i in models if 'mt5' in i.modelId]
# target = [i for i in models_filtered if 'de' in i and "qg" in i]
# pprint(sorted(target))
# if len(target) > 0:
#     input("delete all? >>>")
#     for i in target:
#         api.delete_repo(repo_id=i, repo_type='model')
#
# filt = ModelFilter(author='vocabtrimmer')
# models = api.list_models(filter=filt)
# models_filtered = [i.modelId for i in models if 'mt5-small-trimmed' in i.modelId]
# target = [i for i in models_filtered if i.endswith("qa") and "000" in i]
# pprint(sorted(target))
# if len(target) > 0:
#     input("delete all? >>>")
#     for i in target:
#         api.delete_repo(repo_id=i, repo_type='model')
