import json
import os
import requests
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
os.makedirs("experiments/result", exist_ok=True)
os.makedirs('../metric_files', exist_ok=True)
max_vocab = {"pt": 66554, "it": 67802, "ar": 49871, "fr": 85704, "es": 87080, "de": 91696, "en": 173237}
param_size_full = {"embedding": 192001536, "full": 278295186, "vocab_size": 250002}
param_size_trimmed = {
    5000: {"embedding": 3841536, "full": 89890186, "vocab_size": 5002},
    10000: {"embedding": 7681536, "full": 93735186, "vocab_size": 10002},
    15000: {"embedding": 11521536, "full": 97580186, "vocab_size": 15002},
    30000: {"embedding": 23041536, "full": 109115186, "vocab_size": 30002},
    49871: {"embedding": 38300928, "full": 124394447, "vocab_size": 49871},
    60000: {"embedding": 46081536, "full": 132185186, "vocab_size": 60002},
    90000: {"embedding": 69121536, "full": 155165955, "vocab_size": 90002},
    66554: {"embedding": 51113472, "full": 137223674, "vocab_size": 66554},
    67802: {"embedding": 52071936, "full": 138183386, "vocab_size": 67802},
    85704: {"embedding": 65820672, "full": 151950024, "vocab_size": 85704},
    87080: {"embedding": 66877440, "full": 153008168, "vocab_size": 87080},
    91696: {"embedding": 70422528, "full": 156557872, "vocab_size": 91696},
    173237: {"embedding": 133046016, "full": 219090435, "vocab_size": 173237},
}
param_size_trimmed[param_size_full['vocab_size']] = {"embedding": param_size_full['embedding'], "full": param_size_full['full'], "vocab_size": param_size_full['vocab_size']}


def show_table(table, name):
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
        with open(f'metric_files/{filename}') as f_reader:
            return json.load(f_reader)
    except Exception:
        pass
    print(f'download {url}')
    try:
        with open(f'metric_files/{filename}', "wb") as f_reader:
            r = requests.get(url)
            f_reader.write(r.content)
        with open(f'metric_files/{filename}') as f_reader:
            return json.load(f_reader)
    except Exception:
        return None


full_data = []
for la in sorted(max_vocab.keys()):

    data = {"language": la, 'size': None, "type": "No-Trim"}
    data.update(download(f"{la}.raw.json", url=f"https://huggingface.co/cardiffnlp/xlm-roberta-base-tweet-sentiment-{la}/raw/main/eval.json"))
    full_data.append(data)

    data = {"language": la, 'size': max_vocab[la], "type": "Post-FT (FULL)"}
    data.update(download(f"{la}.json", url=f"https://huggingface.co/vocabtrimmer/xlm-roberta-base-tweet-sentiment-{la}-trimmed-{la}/raw/main/eval.json"))
    full_data.append(data)

    data = {"language": la, 'size': max_vocab[la], "type": "Pre-FT (FULL)"}
    data.update(download(f"{la}.ft_trimmed.json", url=f"https://huggingface.co/vocabtrimmer/xlm-roberta-base-trimmed-{la}-tweet-sentiment-{la}/raw/main/eval.json"))
    full_data.append(data)
    for v_size in [5000, 10000, 15000, 30000, 60000]:
        if v_size > max_vocab[la]:
            continue
        data = {"language": la, 'size': v_size, "type": "Pre-FT"}
        data.update(download(f"{la}.{v_size}.ft_trimmed.json", url=f"https://huggingface.co/vocabtrimmer/xlm-roberta-base-trimmed-{la}-{v_size}-tweet-sentiment-{la}/raw/main/eval.json"))
        full_data.append(data)

        data = {"language": la, 'size': v_size, "type": "Post-FT"}
        data.update(download(f"{la}.{v_size}.json", url=f"https://huggingface.co/vocabtrimmer/xlm-roberta-base-tweet-sentiment-{la}-trimmed-{la}-{v_size}/raw/main/eval.json"))
        full_data.append(data)

df = pd.DataFrame(full_data)[["eval_f1_micro", "eval_recall_micro", "eval_precision_micro",  "eval_f1_macro",  "eval_recall_macro",  "eval_precision_macro",  "eval_accuracy", "language", "size", "type"]]
df[["eval_f1_micro", "eval_recall_micro", "eval_precision_micro",  "eval_f1_macro",  "eval_recall_macro",  "eval_precision_macro",  "eval_accuracy"]] = df[["eval_f1_micro", "eval_recall_micro", "eval_precision_micro",  "eval_f1_macro",  "eval_recall_macro",  "eval_precision_macro",  "eval_accuracy"]] * 100
df['size'] = df['size'].fillna(param_size_full['vocab_size'])
df['param'] = [param_size_trimmed[int(i)]['full'] for i in df['size']]
df.to_csv("experiments/result/sentiment.full.csv", index=False)


df_full = df[["FULL" in i or i == "No-Trim" for i in df['type']]]
df = df[["FULL" not in i for i in df['type']]]
for m in ["eval_f1_micro", "eval_f1_macro"]:

    main_df = None
    for la, g in df.groupby('language'):
        g = g[[m, "size", "param", "type"]]
        g[la] = g.pop(m)
        main_df = g if main_df is None else main_df.merge(g, on=['size', 'type', 'param'], how='outer')

    val_no_trim = main_df[main_df['type'] == 'No-Trim'][[c for c in main_df.columns if c not in ['size', 'type', 'param']]].values
    val = main_df[[c for c in main_df.columns if c not in ['size', 'type', 'param']]].values
    diff = (val.round(1) - val_no_trim.round(1)) >= 0

    def tmp_format(x, y, flag=False):
        if flag:
            return "\textit{" + f"{round(x, 1)}" + "}"
        if str(x) == 'nan':
            return "-"
        if y:
            return "\textbf{" + f"{round(x, 1)}" + "}"
        return f"{round(x, 1)}"

    main_df[[c for c in main_df.columns if c not in ['size', 'type', 'param']]] = [[tmp_format(_v, _d, n == 0) for _v, _d in zip(v, d)] for n, (v, d) in enumerate(zip(val, diff))]
    main_df = main_df.sort_values(by=['type', 'size', 'param'])
    main_df = main_df.round(1)
    main_df = main_df.fillna("-")
    main_df.columns = [c.upper() if len(c) == 2 else c for c in main_df.columns]
    main_df = main_df[['type', 'size', 'param'] + [c for c in main_df.columns if c not in ['type', 'size', 'param']]]
    main_df['Vocab (Param)'] = [f"{int(a / 10 ** 3)}K ({int(c/10**6)}M)" for a, c in zip(main_df.pop('size'), main_df.pop('param'))]
    main_df["Trimming"] = main_df.pop("type")
    main_df_no_trim = main_df[[i == "No-Trim" for i in main_df['Trimming']]]
    langs = [c for c in main_df.columns if c not in ['Vocab (Param)', "Trimming"]]

    main_df = main_df[["Trimming", "Vocab (Param)"] + langs]

    print(f"** metric: {m} **")
    print(show_table(main_df.to_latex(index=False, escape=False), m))

    main_df = df_full[[m, "size", "param", "type", "language"]]
    main_df.index = [i.upper() for i in main_df.pop("language")]
    main_df = (main_df.pivot(columns="type", values=m)).round(1)
    main_df['Post-FT'] = [tmp_format(a, a >= b) for a, b in zip(main_df.pop('Post-FT (FULL)'), main_df["No-Trim"])]
    main_df['Pre-FT'] = [tmp_format(a, a >= b) for a, b in zip(main_df.pop('Pre-FT (FULL)'), main_df["No-Trim"])]
    main_df["size"] = [max_vocab[i.lower()] for i in main_df.index]
    main_df["param"] = [param_size_trimmed[max_vocab[i.lower()]]['full'] for i in main_df.index]
    main_df['Vocab (Param)'] = [f"{int(a / 10 ** 3)}K ({int(c/10**6)}M)" for a, c in zip(main_df.pop('size'), main_df.pop('param'))]
    main_df = main_df[["Vocab (Param)", "Post-FT", "Pre-FT"]]
    main_df["No-Trim"] = [main_df_no_trim[l].values[0].replace("\textit", "") for l in main_df_no_trim[langs]]
    main_df.columns.name = None
    main_df = main_df[["No-Trim", "Pre-FT", "Post-FT", "Vocab (Param)"]]
    print(show_table(main_df.to_latex(escape=False), m))
