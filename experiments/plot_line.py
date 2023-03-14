import os

from itertools import product
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({'font.size': 14})  # must set in top
os.makedirs("experiments/result/figures/line", exist_ok=True)
colors = list(mpl.colormaps['Set1'].colors)
param_size_full_mt5 = {"embedding": 256114688, "full": 300176768, "vocab_size": 250112}
param_size_trimmed_mt5 = {
    15000: {"embedding": 15361024, "full": 59423104, "vocab_size": 15001},
    30000: {"embedding": 30721024, "full": 74783104, "vocab_size": 30001},
    45000: {"embedding": 46081024, "full": 90143104, "vocab_size": 45001},
    60000: {"embedding": 61441024, "full": 105503104, "vocab_size": 60001},
    75000: {"embedding": 76801024, "full": 120863104, "vocab_size": 75001},
    90000: {"embedding": 92161024, "full": 136223104, "vocab_size": 90001},
    105000: {"embedding": 107521024, "full": 151583104, "vocab_size": 105001},
    120000: {"embedding": 122881024, "full": 166943104, "vocab_size": 120001},
    131086: {"embedding": 134232064, "full": 178294144, "vocab_size": 131086},
    125903: {"embedding": 128924672, "full": 172986752, "vocab_size": 125903},
    73356: {"embedding": 75116544, "full": 119178624, "vocab_size": 75001},
    111056: {"embedding": 113721344, "full": 157783424, "vocab_size": 112001},
    147755: {"embedding": 151301120, "full": 195363200, "vocab_size": 148001},
    131105: {"embedding": 134251520, "full": 178313600, "vocab_size": 131106},
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
language_map = {"FR": "French", "ES": "Spanish", "IT": "Italian", "JA": "Japanese", "RU": "Russian", "KO": "Korean",
                "AR": "Arabic", "PT": "Portuguese", "DE": "German"}


def main(data_path: str = "experiments/result/qg.full.csv",
         prefix: str = "mt5_qg",
         metrics=["Bleu_4", "METEOR", "ROUGE_L", "BERTScore", "MoverScore"],
         langs=["FR", "ES", "IT", "JA", "RU", "KO"],
         subplots=(3, 2),
         is_xlm=False):
    df = pd.read_csv(data_path).sort_values(by=["size"])
    if is_xlm:
        param_size_trimmed = param_size_trimmed_xlm
        param_size_full = param_size_full_xlm
    else:
        param_size_trimmed = param_size_trimmed_mt5
        param_size_full = param_size_full_mt5
    df = df[[s in param_size_trimmed or str(s) == 'nan' for s in df['size']]]
    df = df[[l.upper() in langs for l in df['language']]]
    grids = list(product(range(subplots[0]), range(subplots[1])))
    for m in metrics:
        fig, axes = plt.subplots(subplots[0], subplots[1], figsize=(8, 12))
        df_tmp = df[[m, "type", "size", "language"]]
        for n, (la, g) in enumerate(df_tmp.groupby("language")):
            g_full_vocab = pd.DataFrame([{
                m: g[g["type"] == "ft"][m].values[0],
                "Model Size (M)": param_size_full["full"] / 10 ** 6
            }])
            if grids[n][1] == 0:
                ylabel = m.replace("Bleu_4", "BLEU4").replace("AnswerF1Score", "Answer F1").replace("AnswerExactMatch", "Exact Match").replace("eval_f1_micro", "F1").replace("_", "-")
            else:
                ylabel = ''
            if grids[n][0] != 2:
                xlabel = ''
            else:
                xlabel = 'Model Size (M)'
            g_full_vocab.plot.line(xlabel=xlabel, ylabel=ylabel, ax=axes[grids[n][0], grids[n][1]], y=m, x='Model Size (M)', color=colors[0], style="P", markersize=10, label="No-Trim", grid=True)

            g_ft_trimmed = g[g["type"] == "ft_trimmed"]
            g_ft_trimmed["Model Size (M)"] = g_ft_trimmed["size"].apply(lambda x: param_size_trimmed[int(x)]["full"] / 10 ** 6)
            g_ft_trimmed.plot.line(xlabel=xlabel, ylabel=ylabel, ax=axes[grids[n][0], grids[n][1]], y=m, x='Model Size (M)', color=colors[2], style="o:", label="Pre-Trim", grid=True)

            g_trimmed = g[g["type"] == "trimmed"]
            g_trimmed["Model Size (M)"] = g_trimmed["size"].apply(
                lambda x: param_size_trimmed[int(x)]["full"] / 10 ** 6)
            g_trimmed.plot.line(xlabel=xlabel, ylabel=ylabel, ax=axes[grids[n][0], grids[n][1]], y=m,
                                x='Model Size (M)', color=colors[1], style="D--", label="Post-Trim", grid=True)

            if n != len(langs) - 1:
                axes[grids[n][0], grids[n][1]].legend().remove()
            axes[grids[n][0], grids[n][1]].title.set_text(language_map[la.upper()])

        plt.tight_layout()
        plt.savefig(f"experiments/result/figures/line/line.{prefix}.{m}.png", bbox_inches="tight", dpi=600)


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
        metrics=["eval_f1_micro", "eval_recall_micro", "eval_precision_micro"],
        # metrics=["eval_f1_micro"],
        langs=["AR", "IT", "PT", "FR", "ES", "DE"],
        is_xlm=True
    )
