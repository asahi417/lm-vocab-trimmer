
import pandas as pd
from huggingface_hub import ModelFilter, HfApi
model_card = '# Model Card\n'
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

api = HfApi()

# #MLMs
# xlm_v = sorted([i.modelId for i in api.list_models(filter=ModelFilter(author='vocabtrimmer')) if "xlm-v-base-tweet" in i.modelId and 'tweet' not in i.modelId])
# xlm_r = sorted([i.modelId for i in api.list_models(filter=ModelFilter(author='vocabtrimmer')) if "xlm-roberta-base-tweet" in i.modelId and 'tweet' not in i.modelId])

# Sentiment Models (Vanilla)
models = [i.modelId for i in api.list_models(filter=ModelFilter(author='cardiffnlp')) if "base-tweet-sentiment" in i.modelId]
xlm_r = sorted([i for i in models if "xlm-roberta" in i])
xlm_v = sorted([i for i in models if "xlm-v" in i])
card = [{
    "Model": f"[{i}](https://huggingface.co/{i})",
    "Language": i.split("-")[-1],
    "Base Model": "[xlm-roberta-base](https://huggingface.co/xlm-roberta-base)"
    } for i in xlm_r] + [{
    "Model": f"[{i}](https://huggingface.co/{i})",
    "Language": i.split("-")[-1],
    "Base Model": "[facebook/xlm-v-base](https://huggingface.co/facebook/xlm-v-base)"
    } for i in xlm_v]
model_card += f"## Sentiment Models\nLanguage models fine-tuned on [cardiffnlp/tweet_sentiment_multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual). \n\n### Vanilla Fine-tuned Models\n\n{pd.DataFrame(card).to_markdown(index=False)}"

# Sentiment Models (FT)
xlm_v_pre_vt = sorted([i.modelId for i in api.list_models(filter=ModelFilter(author='vocabtrimmer')) if "xlm-v-base-trimmed" in i.modelId and "sentiment" in i.modelId])
xlm_r_pre_vt = sorted([i.modelId for i in api.list_models(filter=ModelFilter(author='vocabtrimmer')) if "xlm-roberta-base-trimmed" in i.modelId and "sentiment" in i.modelId])
xlm_v_post_vt = sorted([i.modelId for i in api.list_models(filter=ModelFilter(author='vocabtrimmer')) if "xlm-v-base-tweet" in i.modelId])
xlm_r_post_vt = sorted([i.modelId for i in api.list_models(filter=ModelFilter(author='vocabtrimmer')) if "xlm-roberta-base-tweet" in i.modelId])
card = [{
    "Model": f"[{i}](https://huggingface.co/{i})",
    "Language": i.split("-")[-1],
    "Base Model": f'[vocabtrimmer/xlm-roberta-base-trimmed-{i.split("-")[-1]}](https://huggingface.co/vocabtrimmer/xlm-roberta-base-trimmed-{i.split("-")[-1]})',
    "VT": "Pre-FT"
    } for i in xlm_r_pre_vt if "0" not in i] + [{
    "Model": f"[{i}](https://huggingface.co/{i})",
    "Language": i.split("-")[-1],
    "Base Model": f'[vocabtrimmer/xlm-v-base-trimmed-{i.split("-")[-1]}](https://huggingface.co/vocabtrimmer/xlm-v-base-trimmed-{i.split("-")[-1]})',
    "VT": "Pre-FT"
    } for i in xlm_v_pre_vt if "0" not in i] + [{
    "Model": f"[{i}](https://huggingface.co/{i})",
    "Language": i.split("-")[-1],
    "Base Model": f'[cardiffnlp/xlm-roberta-base-tweet-sentiment-{i.split("-")[-1]}](https://huggingface.co/cardiffnlp/xlm-roberta-base-tweet-sentiment-{i.split("-")[-1]})',
    "VT": "Post-FT"
    } for i in xlm_r_post_vt if "0" not in i] + [{
    "Model": f"[{i}](https://huggingface.co/{i})",
    "Language": i.split("-")[-1],
    "Base Model": f'[cardiffnlp/xlm-v-base-tweet-sentiment-{i.split("-")[-1]}](https://huggingface.co/cardiffnlp/xlm-roberta-v-tweet-sentiment-{i.split("-")[-1]})',
    "VT": "Post-FT"
    } for i in xlm_v_post_vt if "0" not in i]
model_card += f"\n\n### Vocabulary Trimmed Models\n\n{pd.DataFrame(card).to_markdown(index=False)}"

card = [{
    "Model": f"[{i}](https://huggingface.co/{i})",
    "Language": i.split("-")[-1],
    "Base Model": f'[vocabtrimmer/xlm-roberta-base-trimmed-{i.split("-")[-1]}](https://huggingface.co/vocabtrimmer/xlm-roberta-base-trimmed-{i.split("-")[-1]})',
    "VT": "Pre-FT"
    } for i in xlm_r_pre_vt if "00" in i] + [{
    "Model": f"[{i}](https://huggingface.co/{i})",
    "Language": i.split("-")[-1],
    "Base Model": f'[vocabtrimmer/xlm-v-base-trimmed-{i.split("-")[-1]}](https://huggingface.co/vocabtrimmer/xlm-v-base-trimmed-{i.split("-")[-1]})',
    "VT": "Pre-FT"
    } for i in xlm_v_pre_vt if "00" in i] + [{
    "Model": f"[{i}](https://huggingface.co/{i})",
    "Language": i.split("-")[-2],
    "Base Model": f'[cardiffnlp/xlm-roberta-base-tweet-sentiment-{i.split("-")[-2]}](https://huggingface.co/cardiffnlp/xlm-roberta-base-tweet-sentiment-{i.split("-")[-2]})',
    "VT": "Post-FT"
    } for i in xlm_r_post_vt if i.endswith("0")] + [{
    "Model": f"[{i}](https://huggingface.co/{i})",
    "Language": i.split("-")[-2],
    "Base Model": f'[cardiffnlp/xlm-v-base-tweet-sentiment-{i.split("-")[-2]}](https://huggingface.co/cardiffnlp/xlm-roberta-v-tweet-sentiment-{i.split("-")[-2]})',
    "VT": "Post-FT"
    } for i in xlm_v_post_vt if i.endswith("0")]
df = pd.DataFrame(card)
df['Vocabulary Size'] = [int(i['Model'].split("-")[5] if i['VT'] == 'Pre-FT' else i['Model'].split("-")[-1][:-1]) for _, i in df.iterrows()]
df = df.sort_values(by=["Base Model", "VT", 'Language', 'Vocabulary Size'])
model_card += f"\n\n### Vocabulary Trimmed Models (with top-k vocab)\n\n{df.to_markdown(index=False)}"

with open("model_card.md", 'w') as f:
    f.write(model_card)
