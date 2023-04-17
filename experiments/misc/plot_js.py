"""TODO

Plot: JS-Token Size (line plot for XLM-R/mT5, legend for each language)

Table: Correlation values 8 (4 for each of test/train).
| QG (BS) | QG B4) | QA (F1) | Sentiment (F1) |
Or just mention in the paragraph?
"""
import os
import json
from itertools import chain, product

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
from scipy.spatial import distance
from scipy.stats import pearsonr, spearmanr


cache_dir = 'experiments/cache'
os.makedirs(cache_dir, exist_ok=True)
output_dir = 'experiments/result/analysis'
os.makedirs(output_dir, exist_ok=True)
plt.rcParams.update({'font.size': 14})  # must set in top


def calculate_pvalues(df, corr_type='spearman'):
    df = df.dropna()._get_numeric_data()
    df_cols = pd.DataFrame(columns=df.columns)
    p_values = df_cols.transpose().join(df_cols, how='outer')
    for r in df.columns:
        for c in df.columns:
            if corr_type == 'pearson':
                p_values[r][c] = pearsonr(df[r], df[c])[1]
            elif corr_type == 'spearman':
                p_values[r][c] = spearmanr(df[r], df[c])[1]
            else:
                raise ValueError('unknown: {}'.format(corr_type))
    return p_values


def stats(model,
          language,
          dataset,
          anchor_model,
          dataset_column: str,
          dataset_name=None):

    def tokenize(path, target_model, is_train=True):

        if not os.path.exists(path):
            data = load_dataset(dataset, dataset_name)
            tokenizer = AutoTokenizer.from_pretrained(target_model)
            if is_train:
                text = data['train'][dataset_column] + data['validation'][dataset_column]
            else:
                text = data['test'][dataset_column]
            all_token = [tokenizer.tokenize(t) for t in text]
            i, c = np.unique(list(chain(*all_token)), return_counts=True)
            i, c = i.tolist(), c.tolist()
            freq = dict(zip(i, c))
            with open(path, "w") as f:
                json.dump(freq, f)
        with open(path) as f:
            freq = json.load(f)
        return freq

    def perc(num):
        return round(num * 100, 1)

    # tokenize by the original model
    token_dist = tokenize(
        path=f"{cache_dir}/{os.path.basename(anchor_model)}.{language}.{os.path.basename(dataset)}.{dataset_name}.json",
        target_model=anchor_model)
    token_dist_trim = tokenize(
        path=f"{cache_dir}/{os.path.basename(model)}.{language}.{os.path.basename(dataset)}.{dataset_name}.json",
        target_model=model)
    ids = sorted(list(set(list(token_dist.keys()) + list(token_dist_trim.keys()))))
    js_dist = perc(distance.jensenshannon(
        [token_dist[k] if k in token_dist else 0 for k in ids],
        [token_dist_trim[k] if k in token_dist_trim else 0 for k in ids]))

    token_dist_test = tokenize(
        path=f"{cache_dir}/{os.path.basename(model)}.{language}.{os.path.basename(dataset)}.{dataset_name}.test.json",
        target_model=model,
        is_train=False)
    token_dist_trim_test = tokenize(
        path=f"{cache_dir}/{os.path.basename(anchor_model)}.{language}.{os.path.basename(dataset)}.{dataset_name}.test.json",
        target_model=anchor_model,
        is_train=False)
    ids = sorted(list(set(list(token_dist_test.keys()) + list(token_dist_trim_test.keys()))))
    js_dist_test = perc(distance.jensenshannon(
        [token_dist_test[k] if k in token_dist_test else 0 for k in ids],
        [token_dist_trim_test[k] if k in token_dist_trim_test else 0 for k in ids]))

    token_dist_all = {
        k: int(token_dist[k] if k in token_dist else 0) + int(token_dist_test[k] if k in token_dist_test else 0)
        for k in set(list(token_dist.keys()) + list(token_dist_test.keys()))}
    token_dist_trim_all = {
        k: int(token_dist_test[k] if k in token_dist_test else 0) + int(
            token_dist_trim_test[k] if k in token_dist_trim_test else 0)
        for k in set(list(token_dist_test.keys()) + list(token_dist_trim_test.keys()))}
    ids = sorted(list(set(list(token_dist_all.keys()) + list(token_dist_trim_all.keys()))))
    js_dist_all = perc(distance.jensenshannon(
        [token_dist_all[k] if k in token_dist_all else 0 for k in ids],
        [token_dist_trim_all[k] if k in token_dist_trim_all else 0 for k in ids]))
    return js_dist, js_dist_test, js_dist_all


if __name__ == '__main__':
    full_freq = []
    for la, la_full in zip(["ar", "fr", "it", "es", "de", "pt"], ["arabic", "french", "italian", "spanish", "german", "portuguese"]):
        freq_train = {"la": la.upper(), "split": "train"}
        freq_test = {"la": la.upper(), "split": "test"}
        for size in [None, 15000, 30000, 45000, 60000, 75000, 90000]:
            if la == 'ar' and size is not None and size > 45000:
                continue
            if la in ['fr', 'es'] and size is not None and size > 75000:
                continue
            if la in ['it', 'pt'] and size is not None and size > 60000:
                continue
            train, test, full = stats(
                model=f"vocabtrimmer/xlm-roberta-base-trimmed-{la}{'' if size is None else f'-{size}'}",
                anchor_model="xlm-roberta-base",
                language=la,
                dataset="cardiffnlp/tweet_sentiment_multilingual",
                dataset_column="text",
                dataset_name=la_full)
            freq_train[f"{int(size/1000)}k" if size is not None else "full"] = full
            freq_test[f"{int(size / 1000)}k" if size is not None else "full"] = test

        full_freq.append(freq_train)
        full_freq.append(freq_test)
    df_freq = pd.DataFrame(full_freq)
    df_freq = df_freq[['la'] + [c for c in df_freq.columns if c.endswith("k")] + ["full", "split"]]
    for s, g in df_freq.groupby("split"):
        g.to_csv(f"{output_dir}/js_xlm.{s}.csv", index=False)

    full_freq = []
    for la in ["ja", "fr", "it", "es", "ru", "ko"]:
        freq_train = {"la": la.upper(), "split": "train"}
        freq_test = {"la": la.upper(), "split": "test"}
        for size in [15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000, None]:
            if la == 'ko' and size is not None and size > 60000:
                continue
            if la == 'it' and size is not None and size > 105000:
                continue
            train, test, full = stats(
                model=f"vocabtrimmer/mt5-small-trimmed-{la}{'' if size is None else f'-{size}'}",
                anchor_model="google/mt5-small",
                language=la,
                dataset=f"lmqg/qg_{la}quad",
                dataset_column="paragraph_question")
            freq_train[f"{int(size / 1000)}k" if size is not None else "full"] = full
            freq_test[f"{int(size / 1000)}k" if size is not None else "full"] = test

        full_freq.append(freq_train)
        full_freq.append(freq_test)
    df_freq = pd.DataFrame(full_freq)
    df_freq = df_freq[['la'] + [c for c in df_freq.columns if c.endswith("k")] + ["full", "split"]]
    for s, g in df_freq.groupby("split"):
        g.to_csv(f"{output_dir}/js_mt5.{s}.csv", index=False)

    # compute correlation
    colors = list(mpl.colormaps['Set1'].colors)
    styles = [".", "+"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    grids = list(product(range(2), range(2)))
    corrs = []
    js_div = {"mt5": {}, "xlm": {}}
    for n, (task, metric) in enumerate(zip(["qg", "qg", "qa", "sentiment"], ["BERTScore", "Bleu_4", "AnswerF1Score", "eval_f1_micro"])):
        root_ax = axes[grids[n][0], grids[n][1]]
        for m, s in enumerate(['test', 'train']):
            df_freq_test = pd.read_csv(f"{output_dir}/js_mt5.{s}.csv" if task != 'sentiment' else f"{output_dir}/js_xlm.{s}.csv", index_col=0)
            df_acc = pd.read_csv(f"experiments/result/{task}.full.csv")[[metric, "language", "size", "type"]]
            size_dict = {i: f"{int(i / 1000)}k" if i % 15 == 0 else "full" for i in df_acc['size'] if str(i) != "nan"}
            if s == 'test':
                df_acc = df_acc[df_acc['type'] != "ft_trimmed"]
            else:
                df_acc = df_acc[df_acc['type'] != "trimmed"]
            df_acc["js"] = [df_freq_test[size_dict[i['size']]][i['language'].upper()] if str(i['size']) != 'nan' else 0 for _, i in df_acc.iterrows()]
            norm = {la: i[i['type'] == 'ft'][metric].values[0] for la, i in df_acc.groupby("language")}
            df_acc['metric'] = [i[metric] / norm[i['language']] for _, i in df_acc.iterrows()]
            metric_pretty = metric.replace("AnswerF1Score", "Answer F1").replace("eval_f1_micro", "F1").replace("Bleu_4", "BLEU4")
            task_pretty = task.upper() if task != "sentiment" else "Sentiment"
            df_acc[["js", "metric"]].plot.scatter(
                x="js", y="metric", ylabel=f"{metric_pretty}",
                ax=root_ax,
                xlabel="JS Divergence" if grids[n][0] == 1 else "",
                grid=True, label="Post-Trim" if s == "test" else "Pre-Trim",
                color=colors[m], style=styles[m], marker=styles[m]
            )
            root_ax.title.set_text(task_pretty)
            if n != 3:
                root_ax.legend().remove()

            corrs.append({
                "split": "Post-Trim" if s == "test" else "Pre-Trim",
                "metric": f"{task_pretty} ({metric_pretty})",
                "cor": df_acc[["js", "metric"]].corr("pearson")['js']['metric'],
                "p": calculate_pvalues(df_acc[["js", "metric"]], "pearson")['js']['metric']
            })
            js_div["mt5" if task in ["qa", "qg"] else "xlm"]["Post-Trim" if s == "test" else "Pre-Trim"] = df_acc[['language', 'size', 'js']]

    df = pd.DataFrame(corrs)
    df.to_csv("experiments/result/corr.csv")
    df['score'] = [f"{round(c, 2) if p < 0.005 else '-'}" for c, p in zip(df['cor'], df['p'])]
    print(df)
    df = df.pivot(index="split", columns="metric", values="score")
    print("Correlation")
    print(df.to_latex())
    plt.tight_layout()
    plt.savefig("experiments/result/figures/corr.png", bbox_inches="tight", dpi=600)

    for model in ['mt5', 'xlm']:
        tmp = js_div[model]['Post-Trim'].sort_values(by=['language', 'size'])
        tmp["Post-Trim"] = tmp.pop("js")
        tmp["Pre-Trim"] = js_div[model]['Pre-Trim'].sort_values(by=['language', 'size'])['js'].values
        tmp = tmp.dropna()

        def format_row(_df):
            _size = _df['size'].values[0]
            if _size % 15 != 0:
                return None
            _df.index = _df.pop("language")
            _df = {la.upper(): f"{_df.loc[la]['Pre-Trim']}/{_df.loc[la]['Post-Trim']}" for la in _df.index}
            _df.update({"size": f"{int(_size/10**3)}K"})
            return _df


        df = pd.DataFrame(list(filter(None, [format_row(g) for size, g in tmp.groupby("size")])))
        df.index = df.pop("size").values
        df = df.T
        df = df.fillna('-')
        print("JS Divergence")
        print(df.T.to_latex())
        df.to_csv(f"experiments/result/js.{model}.csv")
