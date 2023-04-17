import os

from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

os.makedirs("result/figures", exist_ok=True)
colors = list(mpl.colormaps['Set1'].colors)
language_map = {"FR": "French", "ES": "Spanish", "IT": "Italian", "JA": "Japanese", "RU": "Russian", "KO": "Korean",
                "AR": "Arabic", "PT": "Portuguese", "DE": "German", "EN": "English"}
plt.rcParams.update({'font.size': 20})  # must set in top

# # QA/QG
# df_full = pd.read_csv("result/qa_qg.csv").sort_values(by=["vocab_size"])
# df_full['language'] = [l.upper() for l in df_full['language']]
# df_full = df_full[df_full['lm'] == "mt5-small"]
#
# metric_qa = "AnswerF1Score"
# y_label_qa = "Ans-F1"
# metric_qg = "METEOR"
# y_label_qg = "METEOR"
# xlabel_main = "Vocab. (K)"
# df_full[metric_qg] = (df_full[metric_qg] * 100).round(1)
#
# # _, axes = plt.subplots(1, 2, figsize=(12, 4))
# _, axes = plt.subplots(7, 2, figsize=(12, 22))
# for m, (la, g_la) in enumerate(df_full.groupby("language")):
#     for n, task in enumerate(["qa", "qg"]):
#         xlabel = xlabel_main if m == len(df_full['language'].unique()) - 1 else ""
#         ylabel = y_label_qa if task == "qa" else y_label_qg
#         metric = metric_qa if task == "qa" else metric_qg
#         ax = axes[m, n]
#         g = g_la[g_la["task"] == task]
#         g_full_vocab = pd.DataFrame([{
#             metric: g[g["type"] == "No-Trim"][metric].values[0], "Vocab. (K)": g[g["type"] == "No-Trim"]["vocab_size"].values[0] / 10 ** 3}])
#         g_full_vocab.plot.line(xlabel=xlabel, ylabel=ylabel, ax=ax, y=metric, x="Vocab. (K)", color=colors[0], style="P:", markersize=10, label="No-Trim", grid=True)
#
#         g_full_vocab = pd.DataFrame([{
#             metric: g[g["type"] == "No-Trim"][metric].values[0], "Vocab. (K)": df_full['vocab_size'].min() / 10 ** 3}] + [{
#             metric: g[g["type"] == "No-Trim"][metric].values[0], "Vocab. (K)": df_full['vocab_size'].max() / 10 ** 3}])
#         g_full_vocab.plot.line(xlabel=xlabel, ylabel=ylabel, ax=ax, y=metric, x="Vocab. (K)", color=colors[0], style=":", markersize=10, label='_nolegend_', grid=True)
#
#         g_ft_trimmed = g[["Pre-FT" in i for i in g["type"]]]
#         g_ft_trimmed["Vocab. (K)"] = [i / 10 ** 3 for i in g_ft_trimmed["vocab_size"]]
#         g_ft_trimmed.plot.line(xlabel=xlabel, ylabel=ylabel, ax=ax, y=metric, x="Vocab. (K)", color=colors[2], style="D-", label="Pre-VT", grid=True)
#
#         g_trimmed = g[["Post-FT" in i for i in g["type"]]]
#         g_trimmed["Vocab. (K)"] = [i / 10 ** 3 for i in g_ft_trimmed["vocab_size"]]
#         g_trimmed.plot.line(xlabel=xlabel, ylabel=ylabel, ax=ax, y=metric, x="Vocab. (K)", color=colors[1], style="o--", label="Post-VT", grid=True)
#         if n != 1 or m != 0:
#             ax.legend().remove()
#         # if n == 0:
#         ax.title.set_text(language_map[la.upper()])
# plt.tight_layout()
# plt.savefig(f"result/figures/line.png", bbox_inches="tight", dpi=400)


# Sentiment
df_full = pd.read_csv("result/sentiment.csv").sort_values(by=["vocab"])
df_full['language'] = [l.upper() for l in df_full['language']]
metric = "eval_f1_macro"
y_label_main = "F1"
xlabel_main = "Vocab. (K)"
df_full[metric] = (df_full[metric] * 100).round(1)

for lm in ["xlm-roberta-base", "xlm-v-base"]:
    df = df_full[df_full['lm'] == lm]
    fig, axes = plt.subplots(4, 2, figsize=(12, 13))
    fig.delaxes(axes[3, 1])
    grids = list(product(range(4), range(2)))
    for m, (la, g) in enumerate(df.groupby("language")):
        xlabel = xlabel_main if m == len(df['language'].unique()) - 1 else ""
        ylabel = y_label_main if grids[m][1] == 0 else ""
        ax = axes[grids[m][0], grids[m][1]]
        g_full_vocab = pd.DataFrame([{
            metric: g[g["type"] == "No-Trim"][metric].values[0], "Vocab. (K)": g[g["type"] == "No-Trim"]["vocab"].values[0] / 10 ** 3}])
        g_full_vocab.plot.line(xlabel=xlabel, ylabel=ylabel, ax=ax, y=metric, x="Vocab. (K)", color=colors[0], style="P:", markersize=10, label="No-Trim", grid=True)

        g_full_vocab = pd.DataFrame([{
            metric: g[g["type"] == "No-Trim"][metric].values[0], "Vocab. (K)": df['vocab'].min() / 10 ** 3}] + [{
            metric: g[g["type"] == "No-Trim"][metric].values[0], "Vocab. (K)": df['vocab'].max() / 10 ** 3}])
        g_full_vocab.plot.line(xlabel=xlabel, ylabel=ylabel, ax=ax, y=metric, x="Vocab. (K)", color=colors[0], style=":", markersize=10, label='_nolegend_', grid=True)

        g_ft_trimmed = g[["Pre-FT" in i for i in g["type"]]]
        g_ft_trimmed["Vocab. (K)"] = [i / 10 ** 3 for i in g_ft_trimmed["vocab"]]
        g_ft_trimmed.plot.line(xlabel=xlabel, ylabel=ylabel, ax=ax, y=metric, x="Vocab. (K)", color=colors[2], style="D-", label="Pre-VT", grid=True)

        g_trimmed = g[["Post-FT" in i for i in g["type"]]]
        g_trimmed["Vocab. (K)"] = [i / 10 ** 3 for i in g_ft_trimmed["vocab"]]
        g_trimmed.plot.line(xlabel=xlabel, ylabel=ylabel, ax=ax, y=metric, x="Vocab. (K)", color=colors[1], style="o--", label="Post-VT", grid=True)
        if m != 1:
            ax.legend().remove()
        ax.title.set_text(language_map[la.upper()])

    plt.tight_layout()
    plt.savefig(f"result/figures/line.{lm}.png", bbox_inches="tight", dpi=400)
