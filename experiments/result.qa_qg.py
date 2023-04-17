import json
import os
import requests
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
os.makedirs("result", exist_ok=True)
os.makedirs('metric_files', exist_ok=True)

max_vocab = {
    "mt5-small": {"ko": 73357, "it": 111056, "ja": 125904, "fr": 131087, "es": 131105, "ru": 147756, 'en': 209329},
    "mbart-large-cc25": {"ko": 46620, "it": 67806, "ja": 77735, "fr": 85707, "es": 87083, "ru": 99641, 'en': 173262}}
param_size_full = {
    "mt5-small": {"embedding": 256114688, "full": 300176768, "vocab_size": 250112},
    "mbart-large-cc25": {"embedding": 512057344, "full": 610852864, "vocab_size": 250028}}
param_size_trimmed = {
    "mt5-small": {
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
        209329: {"embedding": 214352896, "full": 258414976, "vocab_size": 209329}},
    "mbart-large-cc25": {
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
        173262: {"embedding": 354840576, "full": 532244480, "vocab_size": 173262}
    }}

param_size_trimmed["mt5-small"][param_size_full["mt5-small"]['vocab_size']] = {
    "embedding": param_size_full["mt5-small"]['embedding'],
    "full": param_size_full["mt5-small"]['full'],
    "vocab_size": param_size_full["mt5-small"]['vocab_size']}
param_size_trimmed["mbart-large-cc25"][param_size_full["mbart-large-cc25"]['vocab_size']] = {
    "embedding": param_size_full["mt5-small"]['embedding'],
    "full": param_size_full["mt5-small"]['full'],
    "vocab_size": param_size_full["mt5-small"]['vocab_size']}


def show_table(table):
    header = """
\\begin{table*}[t]
\centering
\scalebox{0.75}{\n"""
    footer = """}
\caption{TBA}
\label{tab:tba}
\end{table*}"""
    table = header + table + footer
    return table.replace("NaN", "-")


def download(filename, url):
    try:
        with open(f'metric_files/{filename}') as f_reader:
            return json.load(f_reader)['test']
    except Exception:
        pass
    with open(f'metric_files/{filename}', "wb") as f_reader:
        r = requests.get(url)
        f_reader.write(r.content)
    with open(f'metric_files/{filename}') as f_reader:
        return json.load(f_reader)['test']


full_data = []
for task in ['qg', 'qa']:
    file_prefix = "metric.first.answer.paragraph_question.answer" if task == 'qa' else 'metric.first.sentence.paragraph_answer.question'
    for la in ['en', 'ja', 'ru', 'fr', 'es', 'it', 'ko']:
        data_name = "squad" if la == "en" else f"{la}quad"
        for lm in ['mt5-small', 'mbart-large-cc25']:

            data = {"task": task, "lm": lm, "language": la, 'vocab_size': param_size_full[lm]['vocab_size'], "param_size": param_size_full[lm]['full'], "type": "No-Trim"}
            data.update(download(f"{la}.{task}.{lm}.no_trim.json", url=f"https://huggingface.co/lmqg/{lm}-{data_name}-{task}/raw/main/eval/{file_prefix}.lmqg_qg_{data_name}.default.json"))
            full_data.append(data)

            param_size = param_size_trimmed[lm][param_size_full[lm]['vocab_size']]['full']

            data = {"task": task, "lm": lm, "language": la, 'vocab_size': max_vocab[lm][la], "param_size": param_size, "type": "Post-FT (FULL)"}
            data.update(download(f"{la}.{task}.{lm}.post_ft.json", url=f"https://huggingface.co/vocabtrimmer/{lm}-{data_name}-{task}-trimmed-{la}/raw/main/eval/{file_prefix}.lmqg_qg_{data_name}.default.json"))
            full_data.append(data)

            data = {"task": task, "lm": lm, "language": la, 'vocab_size': max_vocab[lm][la], "param_size": param_size, "type": "Pre-FT (FULL)"}
            data.update(download(f"{la}.{task}.{lm}.pre_ft.json", url=f"https://huggingface.co/vocabtrimmer/{lm}-trimmed-{la}-{data_name}-{task}/raw/main/eval/{file_prefix}.lmqg_qg_{data_name}.default.json"))
            full_data.append(data)

            if lm == 'mt5-small':
                for size in [5000, 10000, 15000, 30000, 60000, 90000, 120000, 150000]:
                    if size in param_size_trimmed[lm]:
                        vocab_size = param_size_trimmed[lm][size]['vocab_size']
                        param_size = param_size_trimmed[lm][size]['full']
                        try:
                            data = {"task": task, "lm": lm, "language": la, 'vocab_size': vocab_size, "param_size": param_size, "type": "Post-FT"}
                            data.update(download(f"{la}.{task}.{lm}.post_ft.{size}.json", url=f"https://huggingface.co/vocabtrimmer/{lm}-{data_name}-{task}-trimmed-{la}-{size}/raw/main/eval/{file_prefix}.lmqg_qg_{data_name}.default.json"))
                            full_data.append(data)
                        except Exception:
                            print(f"https://huggingface.co/vocabtrimmer/{lm}-{data_name}-{task}-trimmed-{la}-{size}/raw/main/eval/{file_prefix}.lmqg_qg_{data_name}.default.json")

                        try:
                            data = {"task": task, "lm": lm, "language": la, 'vocab_size': vocab_size, "param_size": param_size, "type": "Pre-FT"}
                            data.update(download(f"{la}.{task}.{lm}.pre_ft.{size}.json", url=f"https://huggingface.co/vocabtrimmer/{lm}-trimmed-{la}-{size}-{data_name}-{task}/raw/main/eval/{file_prefix}.lmqg_qg_{data_name}.default.json"))
                            full_data.append(data)
                        except Exception:
                            print(f"https://huggingface.co/vocabtrimmer/{lm}-{data_name}-{task}-trimmed-{la}-{size}/raw/main/eval/{file_prefix}.lmqg_qg_{data_name}.default.json")


def emphasize(x, y):
    return f"\textbf{{{x}}}" if x == y else str(x)

df = pd.DataFrame(full_data)
df.to_csv("result/qa_qg.csv", index=False)
output = []
for lm, g in df.groupby('lm'):
    for la, _g in g.groupby("language"):
        v_size = max_vocab[str(lm)][str(la)]
        pretty_lm = "mT5" if lm == "mt5-small" else "mBART"
        tmp = {"lm": pretty_lm, "language": la, "vocab": int(v_size/1000), "model_size": round((param_size_trimmed[str(lm)][v_size]["full"]/param_size_full[str(lm)]["full"]) * 100, 1)}
        for task, __g in _g.groupby('task'):
            __g = __g[["FULL" in t or t == 'No-Trim' for t in __g['type']]]
            __g.index = __g.pop('type')
            __g = __g.sort_index()
            if task == 'qg':
                __g = (__g[['METEOR', 'BERTScore']] * 100).round(1)
                tmp.update({f"{task}.{t}": "/".join([emphasize(i.values[0], __g['METEOR'].max()), emphasize(i.values[1], __g['BERTScore'].max())]) for t, i in __g.iterrows()})
            else:
                __g = __g[['AnswerF1Score', "AnswerExactMatch"]].astype(float).round(1)
                tmp.update({f"{task}.{t}": "/".join([emphasize(i.values[0], __g['AnswerF1Score'].max()), emphasize(i.values[1], __g['AnswerExactMatch'].max())]) for t, i in __g.iterrows()})
        output.append(tmp)
output = pd.DataFrame(output)
print(show_table(output.to_latex(index=False, escape=False)))


df = df[df['lm'] == 'mt5-small']
for task, g in df.groupby('task'):
    for vt_type in ['Pre-FT', 'Post-FT']:
        output = []
        _g = g[[i in [vt_type, "No-Trim"] for i in g['type']]]
        for la, __g in _g.groupby("language"):
            __g['vocab_size'] = (__g['vocab_size'] / 10**3).astype(int)

            if task == 'qg':
                __g[['METEOR', 'BERTScore']] = (__g[['METEOR', 'BERTScore']] * 100).round(1)
                __g[la] = [" / ".join([
                    emphasize(i['METEOR'], __g['METEOR'].max()),
                    emphasize(i['BERTScore'], __g['BERTScore'].max())]
                ) for t, i in __g.iterrows()]
            else:
                __g[['AnswerF1Score', "AnswerExactMatch"]] = __g[['AnswerF1Score', "AnswerExactMatch"]].astype(float).round(1)
                __g[la] = [" / ".join([
                    emphasize(i['AnswerF1Score'], __g['AnswerF1Score'].max()),
                    emphasize(i['AnswerExactMatch'], __g['AnswerExactMatch'].max())]
                ) for t, i in __g.iterrows()]
            __g = __g[[la, 'vocab_size']]
            __g.index = __g.pop('vocab_size')
            output.append(__g.T)
        df_tmp = pd.concat(output)
        df_tmp.index = [i.upper() for i in df_tmp.index]
        df_tmp.columns.name = ""
        df_tmp.columns = [f"{c}K" for c in df_tmp.columns]
        print(task, vt_type)
        print(show_table(df_tmp.to_latex(escape=False)))
