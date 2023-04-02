import json
import os
import requests
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
os.makedirs("experiments/result", exist_ok=True)
os.makedirs('metric_files', exist_ok=True)
max_vocab = {"ko": 73357, "it": 111056, "ja": 125904, "fr": 131087, "es": 131105, "ru": 147756, 'en': 209329}
param_size_full = {"embedding": 256114688, "full": 300176768, "vocab_size": 250112}
param_size_trimmed = {
    5000: {"embedding": 5123072, "full": 49185152, "vocab_size": 5003},
    10000: {"embedding": 10243072, "full": 54305152, "vocab_size": 10003},
    15000: {"embedding": 15361024, "full": 59423104, "vocab_size": 15001},
    30000: {"embedding": 30721024, "full": 74783104, "vocab_size": 30001},
    60000: {"embedding": 61441024, "full": 105503104, "vocab_size": 60001},
    90000: {"embedding": 92161024, "full": 136223104, "vocab_size": 90001},
    120000: {"embedding": 122881024, "full": 166943104, "vocab_size": 120001},
    131087: {"embedding": 134232064, "full": 178294144, "vocab_size": 131086},
    125904: {"embedding": 128924672, "full": 172986752, "vocab_size": 125903},
    73357: {"embedding": 75116544, "full": 119178624, "vocab_size": 75001},
    111056: {"embedding": 113721344, "full": 157783424, "vocab_size": 112001},
    147756: {"embedding": 151301120, "full": 195363200, "vocab_size": 148001},
    131105: {"embedding": 134251520, "full": 178313600, "vocab_size": 131106},
    209329: {"embedding": 214352896, "full": 258414976, "vocab_size": 209329},
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
    filename = f"mt5_qa.{filename}"
    try:
        with open(f'metric_files/{filename}') as f_reader:
            return json.load(f_reader)['test']
    except Exception:
        pass
    print(f'download {url}')
    try:
        with open(f'metric_files/{filename}', "wb") as f_reader:
            r = requests.get(url)
            f_reader.write(r.content)
        with open(f'metric_files/{filename}') as f_reader:
            return json.load(f_reader)['test']
    except Exception:
        return None


full_data = []
for la in ['en', 'ja', 'ru', 'fr', 'es', 'it', 'ko']:
# for la in ['ja', 'ru', 'fr', 'es', 'it', 'ko']:
    data = {"language": la, 'size': None, "type": "No-Trim"}
    data_name = "squad" if la == "en" else f"{la}quad"
    data.update(download(f"{la}.raw.json", url=f"https://huggingface.co/lmqg/mt5-small-{data_name}-qa/raw/main/eval/metric.first.answer.paragraph_question.answer.lmqg_qg_{data_name}.default.json"))
    full_data.append(data)

    data = {"language": la, 'size': max_vocab[la], "type": "Post-FT (FULL)"}
    data.update(download(f"{la}.json", url=f"https://huggingface.co/vocabtrimmer/mt5-small-{data_name}-qa-trimmed-{la}/raw/main/eval/metric.first.answer.paragraph_question.answer.lmqg_qg_{data_name}.default.json"))
    full_data.append(data)

    data = {"language": la, 'size': max_vocab[la], "type": "Pre-FT (FULL)"}
    data.update(download(f"{la}.ft_trimmed.json", url=f"https://huggingface.co/vocabtrimmer/mt5-small-trimmed-{la}-{data_name}-qa/raw/main/eval/metric.first.answer.paragraph_question.answer.lmqg_qg_{data_name}.default.json"))
    full_data.append(data)

    for v_size in [5000, 10000, 15000, 30000, 60000, 90000, 120000]:
        if v_size > max_vocab[la]:
            continue
        data = {"language": la, 'size': v_size, "type": "Pre-FT"}
        data.update(download(f"{la}.{v_size}.ft_trimmed.json", url=f"https://huggingface.co/vocabtrimmer/mt5-small-trimmed-{la}-{v_size}-{data_name}-qa/raw/main/eval/metric.first.answer.paragraph_question.answer.lmqg_qg_{data_name}.default.json"))
        full_data.append(data)

        data = {"language": la, 'size': v_size, "type": "Post-FT"}
        data.update(download(f"{la}.{v_size}.json", url=f"https://huggingface.co/vocabtrimmer/mt5-small-{data_name}-qa-trimmed-{la}-{v_size}/raw/main/eval/metric.first.answer.paragraph_question.answer.lmqg_qg_{data_name}.default.json"))
        full_data.append(data)


df = pd.DataFrame(full_data)[["AnswerF1Score", "AnswerExactMatch", "language", "size", "type"]]
df['size'] = df['size'].fillna(param_size_full['vocab_size'])
df['param'] = [param_size_trimmed[int(i)]['full'] for i in df['size']]
df.to_csv("experiments/result/qa.full.csv", index=False)
df_full = df[["FULL" in i or i == "No-Trim" for i in df['type']]]
df = df[["FULL" not in i for i in df['type']]]


for m in ["AnswerF1Score", "AnswerExactMatch"]:

    main_df = None
    for la, g in df.groupby('language'):
        g = g[[m, "size", "param", "type"]]
        g[la] = g.pop(m)
        main_df = g if main_df is None else main_df.merge(g, on=['size', 'type', 'param'], how='outer')

    val_no_trim = main_df[main_df['type'] == 'No-Trim'][
        [c for c in main_df.columns if c not in ['size', 'type', 'param']]].values
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


    main_df[[c for c in main_df.columns if c not in ['size', 'type', 'param']]] = [
        [tmp_format(_v, _d, n == 0) for _v, _d in zip(v, d)] for n, (v, d) in enumerate(zip(val, diff))]
    main_df = main_df.sort_values(by=['type', 'size', 'param'])
    main_df = main_df.round(1)
    main_df = main_df.fillna("-")
    main_df.columns = [c.upper() if len(c) == 2 else c for c in main_df.columns]
    main_df = main_df[['type', 'size', 'param'] + [c for c in main_df.columns if c not in ['type', 'size', 'param']]]
    main_df['Vocab (Param)'] = [f"{int(a / 10 ** 3)}K ({int(c / 10 ** 6)}M)" for a, c in
                                zip(main_df.pop('size'), main_df.pop('param'))]
    main_df["Trimming"] = main_df.pop("type")
    main_df = main_df[
        ['Vocab (Param)', "Trimming"] + [c for c in main_df.columns if c not in ['Vocab (Param)', "Trimming"]]]

    print(f"** metric: {m} **")
    print(show_table(main_df.to_latex(index=False, escape=False), m))

    main_df = df_full[[m, "size", "param", "type", "language"]]
    main_df.index = [i.upper() for i in main_df.pop("language")]
    main_df = main_df.pivot(columns="type", values=m).round(1)
    main_df['Post-FT'] = [tmp_format(a, a >= b) for a, b in zip(main_df.pop('Post-FT (FULL)'), main_df["No-Trim"])]
    main_df['Pre-FT'] = [tmp_format(a, a >= b) for a, b in zip(main_df.pop('Pre-FT (FULL)'), main_df["No-Trim"])]
    main_df["size"] = [max_vocab[i.lower()] for i in main_df.index]
    main_df["param"] = [param_size_trimmed[max_vocab[i.lower()]]['full'] for i in main_df.index]
    main_df['Vocab (Param)'] = [f"{int(a / 10 ** 3)}K ({int(c / 10 ** 6)}M)" for a, c in
                                zip(main_df.pop('size'), main_df.pop('param'))]
    main_df = main_df[["Vocab (Param)", "Post-FT", "Pre-FT"]]
    print(show_table(main_df.to_latex(escape=False), m))
