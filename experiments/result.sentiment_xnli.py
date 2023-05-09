import json
import os
import requests
import pandas as pd

pd.options.mode.chained_assignment = None  # default='warn'
os.makedirs("result", exist_ok=True)
os.makedirs('metric_files', exist_ok=True)
max_vocab = {
    "xlm-roberta-base": {"en": 173237, "ar": 49871, "fr": 85704, "it": 67802, "de": 91696, "pt": 66554, "es": 87080},
    "xlm-v-base": {"en": 484747, "ar": 92992, "fr": 218448, "it": 184721, "de": 239290, "pt": 181373, "es": 243366}}
param_size_full = {
    "xlm-roberta-base": {"embedding": 192001536, "full": 278295186, "vocab_size": 250002},
    "xlm-v-base": {"embedding": 692451072, "full": 778495491, "vocab_size": 901629}}
param_size_trimmed = {
    "xlm-roberta-base": {
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
        173237: {"embedding": 133046016, "full": 219090435, "vocab_size": 173237}},
    "xlm-v-base": {
        5000: {"embedding": 3841536, "full": 89885955, "vocab_size": 5002},
        10000: {"embedding": 7681536, "full": 93725955, "vocab_size": 10002},
        15000: {"embedding": 11521536, "full": 97565955, "vocab_size": 15002},
        30000: {"embedding": 23041536, "full": 109085955, "vocab_size": 30002},
        60000: {"embedding": 46081536, "full": 132125955, "vocab_size": 60002},
        243366: {"embedding": 186905088, "full": 272949507, "vocab_size": 243366},
        181373: {"embedding": 139294464, "full": 225338883, "vocab_size": 181373},
        92992: {"embedding": 71417856, "full": 157462275, "vocab_size": 92992},
        218448: {"embedding": 167768064, "full": 253812483, "vocab_size": 218448},
        184721: {"embedding": 141865728, "full": 227910147, "vocab_size": 184721},
        239290: {"embedding": 183774720, "full": 269819139, "vocab_size": 239290},
        484747: {"embedding": 372285696, "full": 458330115, "vocab_size": 484747}}}
param_size_trimmed["xlm-roberta-base"][param_size_full["xlm-roberta-base"]['vocab_size']] = {
    "embedding": param_size_full["xlm-roberta-base"]['embedding'],
    "full": param_size_full["xlm-roberta-base"]['full'],
    "vocab_size": param_size_full["xlm-roberta-base"]['vocab_size']}
param_size_trimmed["xlm-v-base"][param_size_full["xlm-v-base"]['vocab_size']] = {
    "embedding": param_size_full["xlm-v-base"]['embedding'],
    "full": param_size_full["xlm-v-base"]['full'],
    "vocab_size": param_size_full["xlm-v-base"]['vocab_size']}


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
            return json.load(f_reader)
    except Exception:
        pass
    print(url)
    with open(f'metric_files/{filename}', "wb") as f_reader:
        r = requests.get(url)
        f_reader.write(r.content)
    with open(f'metric_files/{filename}') as f_reader:
        return json.load(f_reader)


full_data = []
for task in ["tweet-sentiment", "xnli"]:
    for lm in ["xlm-roberta-base", "xlm-v-base"]:
        for la in sorted(max_vocab[lm].keys()):
            if task == 'xnli' and la in ['pt', 'it']:
                continue
            data = {"language": la, 'lm': lm, 'param': param_size_full[lm]['full'], 'vocab': param_size_full[lm]['vocab_size'], "type": "No-Trim", "task": task}
            if task == 'xnli':
                data.update(download(f"{lm}.{la}.json", url=f"https://huggingface.co/vocabtrimmer/{lm}-{task}-{la}/raw/main/eval.json"))
            else:
                data.update(download(f"{lm}.{la}.json", url=f"https://huggingface.co/cardiffnlp/{lm}-{task}-{la}/raw/main/eval.json"))
            full_data.append(data)

            data = {"language": la, 'lm': lm, 'param': param_size_trimmed[lm][max_vocab[lm][la]]["full"], 'vocab': max_vocab[lm][la], "type": "Post-FT (FULL)", "task": task}
            data.update(download(f"{lm}.{la}.post_ft.json", url=f"https://huggingface.co/vocabtrimmer/{lm}-{task}-{la}-trimmed-{la}/raw/main/eval.json"))
            full_data.append(data)

            data = {"language": la, 'lm': lm, 'param': param_size_trimmed[lm][max_vocab[lm][la]]["full"], 'vocab': max_vocab[lm][la], "type": "Pre-FT (FULL)", "task": task}
            data.update(download(f"{lm}.{la}.pre_ft.json", url=f"https://huggingface.co/vocabtrimmer/{lm}-trimmed-{la}-{task}-{la}/raw/main/eval.json"))
            full_data.append(data)
            for v_size in [5000, 10000, 15000, 30000, 60000]:
                try:
                    data = {"language": la, 'lm': lm, 'param': param_size_trimmed[lm][v_size]['full'], 'vocab': param_size_trimmed[lm][v_size]['vocab_size'], "type": "Pre-FT", "task": task}
                    data.update(download(f"{lm}.{la}.pre_ft.{v_size}.json", url=f"https://huggingface.co/vocabtrimmer/{lm}-trimmed-{la}-{v_size}-{task}-{la}/raw/main/eval.json"))
                    full_data.append(data)
                except Exception:
                    print(f"https://huggingface.co/vocabtrimmer/{lm}-trimmed-{la}-{v_size}-{task}-{la}/raw/main/eval.json")
                    pass
                try:
                    data = {"language": la, 'lm': lm, 'param': param_size_trimmed[lm][v_size]['full'], 'vocab': v_size, "type": "Post-FT", "task": task}
                    data.update(download(f"{lm}.{la}.pre_ft.{v_size}.json", url=f"https://huggingface.co/vocabtrimmer/{lm}-{task}-{la}-trimmed-{la}-{v_size}/raw/main/eval.json"))
                    full_data.append(data)
                except Exception:
                    print(f"https://huggingface.co/vocabtrimmer/{lm}-{task}-{la}-trimmed-{la}-{v_size}/raw/main/eval.json")
                    pass

df = pd.DataFrame(full_data)
df.to_csv("result/sentiment_xnli.csv", index=False)

def emphasize(x, y):
    return f"\textbf{{{x}}}" if x == y else str(x)

df = pd.read_csv("result/sentiment_xnli.csv")
output = []
for lm, g in df.groupby('lm'):
    for la, __g in g.groupby("language"):
        v_size = max_vocab[str(lm)][str(la)]
        tmp = {"lm": "XLM-R" if lm == "xlm-roberta-base" else "XLM-V", "language": str(la).upper(), "vocab": int(v_size/1000), "model_size": round((param_size_trimmed[str(lm)][v_size]["full"]/param_size_full[str(lm)]["full"]) * 100, 1)}
        for task, _g in __g.groupby('task'):
            _g = _g[["FULL" in t or t == 'No-Trim' for t in _g['type']]]
            _g.index = _g.pop('type')
            _g = _g.sort_index()
            if task == "xnli":
                _g = (_g[["eval_accuracy"]] * 100).round(1)
                tmp.update({f"{t}.{task}": emphasize(i.values[0], _g["eval_accuracy"].max()) for t, i in _g.iterrows()})
            else:
                _g = (_g[["eval_f1_macro"]] * 100).round(1)
                tmp.update({f"{t}.{task}": emphasize(i.values[0], _g["eval_f1_macro"].max()) for t, i in _g.iterrows()})
        output.append(tmp)
output = pd.DataFrame(output)
print(show_table(output.to_latex(index=False, escape=False)))

input()
for task, df_task in df.groupby('task'):
    # for lm in ["xlm-roberta-base", "xlm-v-base"]:
    df_ = df_task[df_task['lm'] == "xlm-roberta-base"]
    for vt_type in ['Pre-FT', 'Post-FT']:
        output = []
        g = df_[[i in [vt_type, "No-Trim"] for i in df_['type']]]
        for la, _g in g.groupby("language"):
            _g['vocab'] = (_g['vocab'] / 10**3).astype(int)
            _g["eval_f1_macro"] = (_g[["eval_f1_macro"]] * 100).round(1)
            _g[la] = [emphasize(i['eval_f1_macro'], _g['eval_f1_macro'].max()) for t, i in _g.iterrows()]
            _g = _g[[la, 'vocab']]
            _g.index = _g.pop('vocab')
            output.append(_g.T)
        df_tmp = pd.concat(output)
        df_tmp.index = [i.upper() for i in df_tmp.index]
        df_tmp.columns.name = ""
        df_tmp.columns = [f"{c}K" for c in df_tmp.columns]

        print()
        print(task, vt_type)
        print(show_table(df_tmp.to_latex(escape=False)))