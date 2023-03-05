import json
import os
import requests

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

TMP_DIR = 'metric_files'
mt5_max_vocab = {
  "ko":  73357,
  "it": 111056,
  "ja": 125904,
  "fr": 131087,
  "es": 131105,
  "de": 137617,
  "ru": 147756,
}


def download(filename, url):
    try:
        with open(f'{TMP_DIR}/{filename}') as f_reader:
            return json.load(f_reader)
    except Exception:
        pass
    print(f'download {url}')
    try:
        os.makedirs(TMP_DIR, exist_ok=True)
        with open(f'{TMP_DIR}/{filename}', "wb") as f_reader:
            r = requests.get(url)
            f_reader.write(r.content)
        with open(f'{TMP_DIR}/{filename}') as f_reader:
            return json.load(f_reader)
    except Exception:
        return None


full_data = []
for la in ['ja', 'ru', 'fr', 'de', 'es', 'it', 'ko']:

    data = download(
        f"{la}.raw.json",
        url=f"https://huggingface.co/lmqg/mt5-small-{la}quad-qg/raw/main/eval/metric.first.answer.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
    data = data['test']
    data['language'] = la
    data['size'] = None
    data['type'] = "ft"
    full_data.append(data)

    data = download(
        f"{la}.json",
        url=f"https://huggingface.co/vocabtrimmer/mt5-small-{la}quad-qg-trimmed/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
    if data is not None:
        data = data['test']
        data['language'] = la
        data['size'] = mt5_max_vocab[la]
        data['type'] = "trimmed"
    else:
        print(la)
    full_data.append(data)

    data = download(
        f"{la}.ft_trimmed.json",
        url=f"https://huggingface.co/vocabtrimmer/mt5-small-trimmed-{la}-{la}quad-qg/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
    if data is not None:
        data = data['test']
        data['language'] = la
        data['size'] = mt5_max_vocab[la]
        data['type'] = "ft_trimmed"
    else:
        print(la)
    full_data.append(data)

    for v_size in [15000, 45000, 30000, 60000, 75000, 90000, 105000, 120000]:
        if v_size > mt5_max_vocab[la]:
            continue

        data = download(
            f"{la}.{v_size}.ft_trimmed.json",
            url=f"https://huggingface.co/vocabtrimmer/mt5-small-trimmed-{la}-{v_size}-{la}quad-qg/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
        if data is not None:
            data = data['test']
            data['language'] = la
            data['size'] = v_size
            data['type'] = "ft_trimmed"
        else:
            print(la)
        full_data.append(data)

        data = download(
            f"{la}.{v_size}.json",
            url=f"https://huggingface.co/vocabtrimmer/mt5-small-{la}quad-qg-trimmed-{v_size}/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
        if data is not None:
            data = data['test']
            data['language'] = la
            data['size'] = v_size
            data['type'] = "trimmed"
            full_data.append(data)
        else:
            print(la, v_size)


df = pd.DataFrame([i for i in full_data if i is not None])
df = df[["Bleu_4", "BERTScore", "language", "size", "type"]]
df[["Bleu_4", "BERTScore"]] = (df[["Bleu_4", "BERTScore"]] * 100).round(2)
for la, g in df.groupby('language'):
    print(f"\n##{la}##")
    ft = g[g['type'] == 'ft'][["Bleu_4", "BERTScore"]].values
    g = g[g['type'] != 'ft']
    g[["Bleu_4", "BERTScore"]] = g[["Bleu_4", "BERTScore"]] - ft
    print(g.sort_values(by=['size', "type"]))
