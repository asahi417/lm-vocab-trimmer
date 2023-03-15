import json
import os
import requests

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

TMP_DIR = 'metric_files'
max_vocab = {
  "pt": 66554,
  "it": 67802,
  "ar": 49871,
  "fr": 85704,
  "es": 87080,
  "de": 91696
}
param_size_full_xlm = {"embedding": 192001536, "full": 278295186, "vocab_size": 250002}
param_size_trimmed_xlm = {
    15000: {"embedding": 11521536, "full": 97580186, "vocab_size": 15002},
    30000: {"embedding": 23041536, "full": 109115186, "vocab_size": 30002},
    45000: {"embedding": 34561536, "full": 120650186, "vocab_size": 45002},
    49871: {"embedding": 38300928, "full": 124394447, "vocab_size": 49871},
    60000: {"embedding": 46081536, "full": 132185186, "vocab_size": 60002},
    66554: {"embedding": 51113472, "full": 137223674, "vocab_size": 66554},
    67802: {"embedding": 52071936, "full": 138183386, "vocab_size": 67802},
    75000: {"embedding": 57601536, "full": 143720186, "vocab_size": 75002},
    85704: {"embedding": 65820672, "full": 151950024, "vocab_size": 85704},
    87080: {"embedding": 66877440, "full": 153008168, "vocab_size": 87080},
    90000: {"embedding": 69121536, "full": 155255186, "vocab_size": 90002},
    91696: {"embedding": 70422528, "full": 156557872, "vocab_size": 91696},
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
    filename = f"xlm.{filename}"
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
for la in sorted(max_vocab.keys()):

    data = download(
        f"{la}.raw.json",
        url=f"https://huggingface.co/cardiffnlp/xlm-roberta-base-tweet-sentiment-{la}/raw/main/eval.json")
    data['language'] = la
    data['size'] = None
    data['type'] = "ft"
    full_data.append(data)

    data = download(
        f"{la}.json",
        url=f"https://huggingface.co/vocabtrimmer/xlm-roberta-base-tweet-sentiment-{la}-trimmed-{la}/raw/main/eval.json")
    if data is not None:
        data['language'] = la
        data['size'] = max_vocab[la]
        data['type'] = "trimmed"
    else:
        print(la)
    full_data.append(data)

    data = download(
        f"{la}.ft_trimmed.json",
        url=f"https://huggingface.co/vocabtrimmer/xlm-roberta-base-trimmed-{la}-tweet-sentiment-{la}/raw/main/eval.json")
    if data is not None:
        data['language'] = la
        data['size'] = max_vocab[la]
        data['type'] = "ft_trimmed"
    else:
        print(la)
    full_data.append(data)

    for v_size in [5000, 10000, 15000, 30000, 60000, 90000]:
        if v_size > max_vocab[la]:
            continue

        data = download(
            f"{la}.{v_size}.ft_trimmed.json",
            url=f"https://huggingface.co/vocabtrimmer/xlm-roberta-base-trimmed-{la}-{v_size}-tweet-sentiment-{la}/raw/main/eval.json")
        if data is not None:
            data['language'] = la
            data['size'] = v_size
            data['type'] = "ft_trimmed"
        else:
            print(la)
        full_data.append(data)

        data = download(
            f"{la}.{v_size}.json",
            url=f"https://huggingface.co/vocabtrimmer/xlm-roberta-base-tweet-sentiment-{la}-trimmed-{la}-{v_size}/raw/main/eval.json")
        if data is not None:
            data['language'] = la
            data['size'] = v_size
            data['type'] = "trimmed"
            full_data.append(data)
        else:
            print(la, v_size)


df = pd.DataFrame([i for i in full_data if i is not None])
df = df[["eval_f1_micro", "eval_recall_micro", "eval_precision_micro",  "eval_f1_macro",  "eval_recall_macro",  "eval_precision_macro",  "eval_accuracy", "language", "size", "type"]]
df[["eval_f1_micro", "eval_recall_micro", "eval_precision_micro",  "eval_f1_macro",  "eval_recall_macro",  "eval_precision_macro",  "eval_accuracy"]] = (df[["eval_f1_micro", "eval_recall_micro", "eval_precision_micro",  "eval_f1_macro",  "eval_recall_macro",  "eval_precision_macro",  "eval_accuracy"]] * 100).round(2)
os.makedirs("experiments/result", exist_ok=True)
df.to_csv("experiments/result/sentiment.full.csv", index=False)
print(df)

for m in ["eval_f1_micro", "eval_recall_micro", "eval_precision_micro"]:

    main_df = None
    for la, g in df.groupby('language'):
        g = g[[m, "size", "type"]]
        g['param'] = [param_size_trimmed_xlm[int(i)]['full'] if str(i) != 'nan' else 278295186 for i in g['size']]
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

    main_df['type'] = [i.replace("ft_trimmed", "Pre-FT").replace("trimmed", "Post-FT").replace("ft", "No-Trim",) for i in main_df.pop("type")]
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
    main_df.pop("param")
    # main_df = main_df[main_df['size'] != 'Full']
    print(f"** metric: {m} **")
    print(show_table(main_df.to_latex(index=False, escape=False), m))