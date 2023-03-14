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
  "ru": 147756,
}
param_size_trimmed_mt5 = {
    15000: {"embedding": 15361024, "full": 59423104, "vocab_size": 15001},
    30000: {"embedding": 30721024, "full": 74783104, "vocab_size": 30001},
    45000: {"embedding": 46081024, "full": 90143104, "vocab_size": 45001},
    60000: {"embedding": 61441024, "full": 105503104, "vocab_size": 60001},
    75000: {"embedding": 76801024, "full": 120863104, "vocab_size": 75001},
    90000: {"embedding": 92161024, "full": 136223104, "vocab_size": 90001},
    105000: {"embedding": 107521024, "full": 151583104, "vocab_size": 105001},
    120000: {"embedding": 122881024, "full": 166943104, "vocab_size": 120001},
    131087: {"embedding": 134232064, "full": 178294144, "vocab_size": 131086},
    125904: {"embedding": 128924672, "full": 172986752, "vocab_size": 125903},
    73357: {"embedding": 75116544, "full": 119178624, "vocab_size": 75001},
    111056: {"embedding": 113721344, "full": 157783424, "vocab_size": 112001},
    147756: {"embedding": 151301120, "full": 195363200, "vocab_size": 148001},
    131105: {"embedding": 134251520, "full": 178313600, "vocab_size": 131106},
}


def show_table(table, name):
    table = table.replace("{llllllll}", "{@{}l@{\hspace{5pt}}l@{\hspace{5pt}}r@{\hspace{5pt}}r@{\hspace{5pt}}r@{\hspace{5pt}}r@{\hspace{5pt}}r@{\hspace{5pt}}r@{}}")
    header = """
\\begin{table}[t]
\centering
\scalebox{0.75}{\n"""
    footer = """}
\caption{TBA}
\label{tab:tba}
\end{table}""".replace("TBA", name)
    return header + table + footer



def download(filename, url):
    filename = f"mt5_qg.{filename}"
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
for la in ['ja', 'ru', 'fr', 'es', 'it', 'ko']:

    data = download(
        f"{la}.raw.json",
        url=f"https://huggingface.co/lmqg/mt5-small-{la}quad-qg/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
    data = data['test']
    data['language'] = la
    data['size'] = None
    data['type'] = "ft"
    full_data.append(data)

    data = download(
        f"{la}.json",
        url=f"https://huggingface.co/vocabtrimmer/mt5-small-{la}quad-qg-trimmed-{la}/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
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
            url=f"https://huggingface.co/vocabtrimmer/mt5-small-{la}quad-qg-trimmed-{la}-{v_size}/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
        if data is not None:
            data = data['test']
            data['language'] = la
            data['size'] = v_size
            data['type'] = "trimmed"
            full_data.append(data)
        else:
            print(la, v_size)


df = pd.DataFrame([i for i in full_data if i is not None])
df = df[['Bleu_4', 'METEOR', 'ROUGE_L', 'BERTScore', 'MoverScore', "language", "size", "type"]]
df[['Bleu_4', 'METEOR', 'ROUGE_L', 'BERTScore', 'MoverScore']] = (df[['Bleu_4', 'METEOR', 'ROUGE_L', 'BERTScore', 'MoverScore']] * 100).round(2)
os.makedirs("experiments/result", exist_ok=True)
df.to_csv("experiments/result/qg.full.csv", index=False)
for m in ['Bleu_4', 'METEOR', 'ROUGE_L', 'BERTScore', 'MoverScore']:

    main_df = None
    for la, g in df.groupby('language'):
        g = g[[m, "size", "type"]]
        g['param'] = [param_size_trimmed_mt5[int(i)]['full'] if str(i) != 'nan' else 300176768 for i in g['size']]
        g[la] = g.pop(m)
        g['size'] = [i if i % 15 == 0 else 250*10**3 for i in g['size']]
        if main_df is None:
            main_df = g
        else:
            main_df = main_df.merge(g, on=['size', 'type', 'param'], how='outer')
    val_no_trim = main_df[main_df['type'] == 'ft'][[c for c in main_df.columns if c not in ['size', 'type', 'param']]].values
    val = main_df[[c for c in main_df.columns if c not in ['size', 'type', 'param']]].values
    diff = val - val_no_trim > 0

    def tmp_format(x, y):
        if str(x) == 'nan':
            return "-"
        if y:
            return "\textbf{" + f"{round(x, 1)}" + "}"
        return f"{round(x, 1)}"

    main_df[[c for c in main_df.columns if c not in ['size', 'type', 'param']]] = [[tmp_format(_v, _d) for _v, _d in zip(v, d)] for v, d in zip(val, diff)]

    main_df['type'] = [i.replace("ft_trimmed", "Pre-Trim").replace("trimmed", "Post-Trim").replace("ft", "No-Trim",) for i in main_df.pop("type")]
    main_df = main_df.sort_values(by=['type', 'size', 'param'])

    main_df = main_df.round(1)
    main_df = main_df.fillna("-")
    main_df.columns = [c.upper() if len(c) == 2 else c for c in main_df.columns]
    main_df = main_df[['type', 'size', 'param'] + [c for c in main_df.columns if c not in ['type', 'size', 'param']]]

    def tmp_format(x, y, z):
        if y == "No-Trim":
            return f"{int(x / 10 ** 3)}K ({int(z/10**6)}M)"
        if x == 250000.0:
            return "Full"
        return f"{int(x / 10 ** 3)}K ({int(z/10**6)}M)"

    main_df['size'] = [tmp_format(a, b, c) for a, b, c in zip(main_df['size'], main_df['type'], main_df['param'])]
    # main_df = main_df.sort_values(by=['type', 'param'])
    main_df.pop("param")
    main_df = main_df[main_df['size'] != 'Full']
    print(f"** metric: {m} **")
    print(show_table(main_df.to_latex(index=False, escape=False), m))
