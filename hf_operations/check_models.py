from pprint import pprint
from huggingface_hub import ModelFilter, HfApi

api = HfApi()
filt = ModelFilter(author='vocabtrimmer')
models = api.list_models(filter=filt)
models_filtered = [i.modelId for i in models if "qg" not in i.modelId and "xlm" not in i.modelId]


for size in ['small']:

    for la in ['ja', 'ko', 'ru', 'fr', 'de', 'es', 'it']:
        print()
        print(f"## {la}/{size}")
        print("- Foundation Model")
        pprint(sorted(
            [i for i in models_filtered if 'qa' not in i and f'mt5-{size}' in i and la in i],
            key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isnumeric() else 0))
        print("- FT QG")
        pprint(sorted(
            [i for i in models_filtered if 'quad' in i and f'mt5-{size}' in i and i.endswith("-qa") and la in i],
            key=lambda x: int(x.split('-')[-3]) if x.split('-')[-3].isnumeric() else 0)
        )
        print("- Trimmed QG")
        pprint(sorted(
            [i for i in models_filtered if 'qa-trimmed' in i and f'mt5-{size}' in i and la in i],
            key=lambda x: int(x.split('-')[-1]) if x.split('-')[-1].isnumeric() else 0)
        )
