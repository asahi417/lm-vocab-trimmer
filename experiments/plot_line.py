import os

from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 14})  # must set in top
os.makedirs("experiments/result/figures/line", exist_ok=True)
colors = list(mpl.colormaps['Set1'].colors)
language_map = {"FR": "French", "ES": "Spanish", "IT": "Italian", "JA": "Japanese", "RU": "Russian", "KO": "Korean",
                "AR": "Arabic", "PT": "Portuguese", "DE": "German"}


def main(data_path: str = "experiments/result/qg.full.csv",
         prefix: str = "mt5_qg",
         metrics=["Bleu_4", "METEOR", "ROUGE_L", "BERTScore", "MoverScore"],
         langs=["FR", "ES", "IT", "JA", "RU", "KO"],
         subplots=(3, 2),
         is_xlm=False):
    df = pd.read_csv(data_path).sort_values(by=["size"])
    df = df[[l.upper() in langs for l in df['language']]]
    df["param"]
    grids = list(product(range(subplots[0]), range(subplots[1])))
    for m in metrics:
        fig, axes = plt.subplots(subplots[0], subplots[1], figsize=(8, 12))
        df_tmp = df[[m, "type", "size", "language", "param"]]
        for n, (la, g) in enumerate(df_tmp.groupby("language")):
            g_full_vocab = pd.DataFrame([{
                m: g[g["type"] == "No-Trim"][m].values[0],
                "Model Size (M)": g[g["type"] == "No-Trim"]["param"].values[0] / 10 ** 6
            }])
            ylabel = m.replace("Bleu_4", "BLEU4").replace("AnswerF1Score", "Answer F1").replace("AnswerExactMatch", "Exact Match").replace("eval_f1_micro", "F1").replace("_", "-") if grids[n][1] == 0 else ''
            xlabel = '' if grids[n][0] != 2 else 'Model Size (M)'
            g_full_vocab.plot.line(xlabel=xlabel, ylabel=ylabel, ax=axes[grids[n][0], grids[n][1]], y=m, x='Model Size (M)', color=colors[0], style="P:", markersize=10, label="No-Trim", grid=True)

            g_full_vocab = pd.DataFrame([{
                m: g[g["type"] == "No-Trim"][m].values[0],
                "Model Size (M)": df_tmp['param'].min() / 10 ** 6
            }] + [{
                m: g[g["type"] == "No-Trim"][m].values[0],
                "Model Size (M)": df_tmp['param'].max() / 10 ** 6
            }])
            print(g_full_vocab)
            g_full_vocab.plot.line(xlabel=xlabel, ylabel=ylabel, ax=axes[grids[n][0], grids[n][1]], y=m, x='Model Size (M)', color=colors[0], style=":", markersize=10, label='_nolegend_', grid=True)

            g_ft_trimmed = g[["Pre-FT" in i for i in g["type"]]]
            g_ft_trimmed["Model Size (M)"] = [i / 10 ** 6 for i in g_ft_trimmed["param"]]
            g_ft_trimmed.plot.line(xlabel=xlabel, ylabel=ylabel, ax=axes[grids[n][0], grids[n][1]], y=m, x='Model Size (M)', color=colors[2], style="o--", label="Pre-Trim", grid=True)

            g_trimmed = g[["Post-FT" in i for i in g["type"]]]
            g_trimmed["Model Size (M)"] = [i / 10 ** 6 for i in g_ft_trimmed["param"]]
            g_trimmed.plot.line(xlabel=xlabel, ylabel=ylabel, ax=axes[grids[n][0], grids[n][1]], y=m,
                                x='Model Size (M)', color=colors[1], style="D-", label="Post-Trim", grid=True)

            if n != len(langs) - 1:
                axes[grids[n][0], grids[n][1]].legend().remove()
            axes[grids[n][0], grids[n][1]].title.set_text(language_map[la.upper()])

        plt.tight_layout()
        plt.savefig(f"experiments/result/figures/line/line.{prefix}.{m}.png", bbox_inches="tight", dpi=600)


if __name__ == '__main__':
    # main(
    #     data_path="experiments/result/qg.full.csv",
    #     prefix="mt5_qg",
    #     metrics=["Bleu_4", "METEOR", "ROUGE_L", "BERTScore", "MoverScore"],
    #     langs=["FR", "ES", "IT", "JA", "RU", "KO"]
    # )
    # main(
    #     data_path="experiments/result/qa.full.csv",
    #     prefix="mt5_qa",
    #     metrics=["AnswerF1Score", "AnswerExactMatch"],
    #     langs=["FR", "ES", "IT", "JA", "RU", "KO"]
    # )
    main(
        data_path="experiments/result/sentiment.full.csv",
        prefix="xlm_r_sentiment",
        # metrics=["eval_f1_micro", "eval_recall_micro", "eval_precision_micro"],
        metrics=["eval_f1_micro"],
        langs=["AR", "IT", "PT", "FR", "ES", "DE"],
        is_xlm=True
    )
