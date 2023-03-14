import os

import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 18})  # must set in top
os.makedirs("experiments/result/figures", exist_ok=True)


# plot
def pretty_type(_type, _size):
    def pretty(num):
        return "{:,}k".format(int(num/1000))

    if _type == "ft":
        return "Full Vocabulary"
    if _type == 'trimmed':
        return f"FT->Trim ({pretty(_size)})"
    if _type == 'ft_trimmed':
        return f"Trim->FT ({pretty(_size)})"
    raise ValueError("NOOOOO")


def main(data_path: str = "experiments/result/qg.full.csv",
         prefix: str = "mt5_qg",
         metrics=["Bleu_4", "METEOR", "ROUGE_L", "BERTScore", "MoverScore"],
         langs=["FR", "ES", "IT", "JA", "RU", "KO"],
         legend_out=True):
    df = pd.read_csv(data_path).sort_values(by=["size"])
    size = [15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000]
    df = df[[s in size or str(s) == 'nan' for s in df['size']]]

    for m in metrics:
        if m == "Bleu_4":
            ylim = [0, 32]
        elif m == "METEOR":
            ylim = [0, 32]
        elif m == "ROUGE_L":
            ylim = [0, 55]
        elif m == "BERTScore":
            ylim = [55, 90]
        elif m == "MoverScore":
            ylim = [45, 85]
        elif m == "AnswerExactMatch":
            ylim = [0, 80]
        elif m == "AnswerF1Score":
            ylim = [0, 80]
        elif m in ["eval_f1_micro", "eval_recall_micro", "eval_precision_micro"]:
            ylim = [55, 75]
        df_tmp = df[[m, "type", "size", "language"]]
        t = df_tmp.pop("type")
        s = df_tmp.pop("size")
        df_tmp['type'] = [pretty_type(_t, _s) for _s, _t in zip(s, t)]
        tmp_out = None
        for l, g in df_tmp.groupby("language"):
            g.pop("language")
            g.index = g.pop("type")
            g[l.upper()] = g.pop(m)
            if tmp_out is None:
                tmp_out = g
            else:
                tmp_out = tmp_out.join(g, how="outer")
        tmp_out = tmp_out.T[
            [i for i in tmp_out.index if "Trim->FT" in i] + [i for i in tmp_out.index if "FT->Trim" in i] + ['Full Vocabulary']].T
        tmp_out = tmp_out[langs].T
        for target in ["FT->Trim", "Trim->FT"]:
            # add_legend = target == "FT->Trim"
            add_legend = True
            tmp_tmp_out = tmp_out[[c for c in tmp_out.columns if target in c or c == "Full Vocabulary"]]
            # ax = tmp_tmp_out.plot.bar(rot=0, figsize=(12, 6), colormap='tab10', width=0.85, legend=add_legend)
            ax = tmp_tmp_out.plot.bar(rot=0, figsize=(12, 6), colormap='viridis', width=0.85, legend=add_legend)

            # ylim
            # ax.set_ylim(min(0, tmp_tmp_out.min().min() - 2), max(100, tmp_tmp_out.max().max() + 2))
            ax.set_ylim(ylim[0], ylim[1])

            # hatch bar
            bars = ax.patches
            # hatches = ''.join(h * len(tmp_tmp_out) for h in 'x/O.')
            patterns = ('', '-', '+', 'x', '/', '//', 'O', 'o', '\\', '\\\\')
            hatches = [h for h in patterns for _ in range(len(tmp_tmp_out))]
            for bar, hatch in zip(bars, hatches):
                bar.set_hatch(hatch)

            # legend
            if add_legend:
                handles, labels = ax.get_legend_handles_labels()
                if legend_out:
                    ax.legend(handles=handles, labels=labels, loc='center left', bbox_to_anchor=(1, 0.5))
                else:
                    ax.legend(handles=handles, labels=labels)

            plt.tight_layout()
            plt.savefig(f"experiments/result/figures/bar.{prefix}.{m}.{target}.png", bbox_inches="tight", dpi=600)


if __name__ == '__main__':
    main(
        data_path="experiments/result/qg.full.csv",
        prefix="mt5_qg",
        metrics=["Bleu_4", "METEOR", "ROUGE_L", "BERTScore", "MoverScore"],
        langs=["FR", "ES", "IT", "JA", "RU", "KO"]
    )
    main(
        data_path="experiments/result/qa.full.csv",
        prefix="mt5_qa",
        metrics=["AnswerF1Score", "AnswerExactMatch"],
        langs=["FR", "ES", "IT", "JA", "RU", "KO"]
    )
    main(
        data_path="experiments/result/sentiment.full.csv",
        prefix="xlm_r_sentiment",
        # metrics=["eval_f1_micro", "eval_recall_micro", "eval_precision_micro"],
        metrics=["eval_f1_micro"],
        langs=["IT", "AR", "PT", "FR", "ES", "DE"],
        legend_out=False
    )
