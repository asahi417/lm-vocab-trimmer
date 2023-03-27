import json
import os
import requests

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

TMP_DIR = 'metric_files'
SKIP_LOADING = True
mbart_max_vocab = {
  "ko": 46620,
  "it": 67806,
  "ja": 77735,
  "fr": 85707,
  "es": 87083,
  "ru": 99641,
}

param_size_trimmed_mt5 = {
    5000: {"embedding": 10248192, "full": 359948288, "vocab_size": 5004},
    10000: {"embedding": 20488192, "full": 365068288, "vocab_size": 10004},
    15000: {"embedding": 30728192, "full": 370188288, "vocab_size": 15004},
    30000: {"embedding": 61448192, "full": 385548288, "vocab_size": 30004},
    60000: {"embedding": 122886144, "full": 416267264, "vocab_size": 60003},
    90000: {"embedding": 184326144, "full": 446987264, "vocab_size": 90003},
    77735: {"embedding": 159201280, "full": 434424832, "vocab_size": 77735},
    87083: {"embedding": 178345984, "full": 443997184, "vocab_size": 87083},
    67806: {"embedding": 138866688, "full": 424257536, "vocab_size": 67806},
    85707: {"embedding": 175527936, "full": 442588160, "vocab_size": 85707},
    99641: {"embedding": 204064768, "full": 456856576, "vocab_size": 99641},
    46620: {"embedding": 95477760, "full": 402563072, "vocab_size": 46620},
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
    filename = f"mbart_qg.{filename}"
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

if not SKIP_LOADING:

    full_data = []
    for la in ['ja', 'ru', 'fr', 'es', 'it', 'ko']:

        data = download(
            f"{la}.raw.json",
            url=f"https://huggingface.co/lmqg/mbart-large-cc25-{la}quad-qg/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
        data = data['test']
        data['language'] = la
        data['size'] = None
        data['type'] = "ft"
        full_data.append(data)

        data = download(
            f"{la}.json",
            url=f"https://huggingface.co/vocabtrimmer/mbart-large-cc25-{la}quad-qg-trimmed-{la}/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
        if data is not None:
            data = data['test']
            data['language'] = la
            data['size'] = mbart_max_vocab[la]
            data['type'] = "trimmed"
        else:
            print(la)
        full_data.append(data)

        data = download(
            f"{la}.ft_trimmed.json",
            url=f"https://huggingface.co/vocabtrimmer/mbart-large-cc25-trimmed-{la}-{la}quad-qg/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
        if data is not None:
            data = data['test']
            data['language'] = la
            data['size'] = mbart_max_vocab[la]
            data['type'] = "ft_trimmed"
        else:
            print(la)
        full_data.append(data)

        for v_size in [5000, 10000, 15000, 30000, 60000, 90000, 120000]:
            if v_size > mbart_max_vocab[la]:
                continue

            data = download(
                f"{la}.{v_size}.ft_trimmed.json",
                url=f"https://huggingface.co/vocabtrimmer/mbart-large-cc25-trimmed-{la}-{v_size}-{la}quad-qg/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
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
                url=f"https://huggingface.co/vocabtrimmer/mbart-large-cc25-{la}quad-qg-trimmed-{la}-{v_size}/raw/main/eval/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json")
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
    df.to_csv("experiments/result/qa_mbart.full.csv", index=False)

# remove the full vocab trimming result
df = pd.read_csv("experiments/result/qa_mbart.full.csv")
print(df)
df = df[[int(i) not in mbart_max_vocab.values() if str(i) != 'nan' else True for i in df['size']]]

for m in ['Bleu_4', 'METEOR', 'ROUGE_L', 'BERTScore', 'MoverScore']:

    main_df = None
    for la, g in df.groupby('language'):
        g = g[[m, "size", "type"]]
        g['param'] = [param_size_trimmed_mt5[int(i)]['full'] if str(i) != 'nan' else 610852864 for i in g['size']]
        g[la] = g.pop(m)
        g['size'] = [i if i % 5 == 0 else 250*10**3 for i in g['size']]
        if main_df is None:
            main_df = g
        else:
            main_df = main_df.merge(g, on=['size', 'type', 'param'], how='outer')
    val_no_trim = main_df[main_df['type'] == 'ft'][[c for c in main_df.columns if c not in ['size', 'type', 'param']]].values
    val = main_df[[c for c in main_df.columns if c not in ['size', 'type', 'param']]].values
    diff = (val.round(1) - val_no_trim.round(1)) >= 0

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
    # main_df = main_df.sort_values(by=['type', 'param'])
    main_df.pop("param")
    main_df = main_df[main_df['size'] != 'Full']
    print(f"** metric: {m} **")
    print(show_table(main_df.to_latex(index=False, escape=False), m))
