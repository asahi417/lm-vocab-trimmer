import pandas as pd
import matplotlib.pyplot as plt


def pretty(num): return "{:,}".format(int(num))

plt.rcParams.update({'font.size': 14})  # must set in top
model_size = {"xlm": 86048650, "mbart": 349700096, "mt5": 44062080}
dims = {"xlm": 768, "mbart": 2048, "mt5": 1024}
param_size_trimmed_mbart = {  # 349700096
    5000: {"embedding": 10248192, "full": 359948288, "vocab_size": 5004},
    10000: {"embedding": 20488192, "full": 365068288, "vocab_size": 10004},
    15000: {"embedding": 30728192, "full": 370188288, "vocab_size": 15004},
    30000: {"embedding": 61448192, "full": 385548288, "vocab_size": 30004},
    60000: {"embedding": 122886144, "full": 416267264, "vocab_size": 60003},
    90000: {"embedding": 184326144, "full": 446987264, "vocab_size": 90003},
    120000: {"embedding": 245766144, "full": 508427264, "vocab_size": 120003},
    "full": {"embedding": 512057344, "full": 610852864, "vocab_size": 250028}}
param_size_trimmed_mt5 = {  # 44062080
    5000: {"embedding": 5123072, "full": 49185152, "vocab_size": 5003},
    10000: {"embedding": 10243072, "full": 54305152, "vocab_size": 10003},
    15000: {"embedding": 15361024, "full": 59423104, "vocab_size": 15001},
    30000: {"embedding": 30721024, "full": 74783104, "vocab_size": 30001},
    60000: {"embedding": 61441024, "full": 105503104, "vocab_size": 60001},
    90000: {"embedding": 92161024, "full": 136223104, "vocab_size": 90001},
    120000: {"embedding": 122881024, "full": 166943104, "vocab_size": 120001},
    "full": {"embedding": 256114688, "full": 300176768, "vocab_size": 250112}}
param_size_trimmed_xlm = {  # 86048650
    5000: {"embedding": 3841536, "full": 89890186, "vocab_size": 5002},
    10000: {"embedding": 7681536, "full": 93735186, "vocab_size": 10002},
    15000: {"embedding": 11521536, "full": 97580186, "vocab_size": 15002},
    30000: {"embedding": 23041536, "full": 109115186, "vocab_size": 30002},
    60000: {"embedding": 46081536, "full": 132185186, "vocab_size": 60002},
    90000: {"embedding": 69121536, "full": 155255186, "vocab_size": 90002},
    120000: {"embedding": 92161536, "full": 178215186, "vocab_size": 120002},
    "full": {"embedding": 192001536, "full": 278295186, "vocab_size": 250002}}
mt5_max_vocab = {"ko":  73356, "it": 111056, "ja": 125903, "fr": 131086, "es": 131105, "ru": 147755}
mbart_max_vocab = {"ko": 46620, "it": 67806, "ja": 77735, "fr": 85707, "es": 87083, "ru": 99641}
xlm_max_vocab = {"pt": 66554, "it": 67802, "ar": 49871, "fr": 85704, "es": 87080, "de": 91696}
language_map = {"ja": "Japanese", "ko": "Korean", "es": "Spanish", "ar": "Arabic", "fr": "French", "ru": "Russian", "it": "Italian", "de": "German", "pt": "Portuguese"}
xlm = [{
    "Model": "XLM-R\textsubscript{BASE}",
    "Language": language_map[l],
    "Vocab": xlm_max_vocab[l],
    "Param": model_size["xlm"] + xlm_max_vocab[l] * dims["xlm"]
} for l in xlm_max_vocab]
mt5 = [{
    "Model": "mT5\textsubscript{SMALL}",
    "Language": language_map[l],
    "Vocab": mt5_max_vocab[l],
    "Param": model_size["mt5"] + mt5_max_vocab[l] * dims["mt5"]
} for l in mt5_max_vocab]
mbart = [{
    "Model": "mBART\textsubscript{LARGE}",
    "Language": language_map[l],
    "Vocab": mbart_max_vocab[l],
    "Param": model_size['mbart'] + mbart_max_vocab[l] * dims["mbart"]
} for l in mbart_max_vocab]

df_1 = pd.DataFrame(xlm + mt5 + mbart)
df_1['Vocabulary Size (K)'] = [f"{round(i/10**3)}K" for i in df_1.pop("Vocab")]
df_1['Parameter Size (M)'] = [f"{round(i/10**6)}M" for i in df_1.pop("Param")]
print("- Table 1")
print(df_1.to_latex(index=False, escape=False))

output_tmp = []
for s in [5000, 10000, 15000, 30000, 60000, 90000, 120000, None]:
    if s is None:
        output_tmp.append({
            "Vocab": 250,
            "Param (XLM-R)": param_size_trimmed_xlm['full']["full"] / 10**6,
            "Param (mT5)": param_size_trimmed_mt5['full']["full"] / 10**6,
            "Param (mBART)": param_size_trimmed_mbart['full']["full"] / 10**6
        })
    else:
        output_tmp.append({
            "Vocab": s/10**3,
            "Param (XLM-R)": param_size_trimmed_xlm[s]['full'] / 10**6,
            "Param (mT5)": param_size_trimmed_mt5[s]['full'] / 10**6,
            "Param (mBART)": param_size_trimmed_mbart[s]["full"] / 10**6,
        })
df_2 = pd.DataFrame(output_tmp)
df_2.index = df_2.pop("Vocab")
print("- Table 2")
print(df_2.to_latex(escape=False))
