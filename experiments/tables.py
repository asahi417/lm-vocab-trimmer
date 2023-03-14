import pandas as pd

def pretty(num): return "{:,}".format(int(num))


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
mt5_max_vocab = {
  "ko":  73356,
  "it": 111056,
  "ja": 125903,
  "fr": 131086,
  "es": 131105,
  "ru": 147755,
}
language_map = {
    "ja": "Japanese",
    "ko": "Korean",
    "es": "Spanish",
    "ar": "Arabic",
    "fr": "French",
    "ru": "Russian",
    "it": "Italian",
    "de": "German",
    "pt": "Portuguese"
}
mt5 = [{
    "Model": "mT5\textsubscript{SMALL}",
    "Language": language_map[l],
    "Vocab": pretty(mt5_max_vocab[l]/10**3)+"K",
    "Param": pretty(param_size_trimmed_mt5[mt5_max_vocab[l]]["full"]/10**6)+"M"
} for l in mt5_max_vocab] + [{
    "Model": "mT5\textsubscript{SMALL}",
    "Language": "No Trim",
    "Vocab": pretty(param_size_full_mt5["vocab_size"]/10**3)+"K",
    "Param": pretty(param_size_full_mt5["full"]/10**6)+"M"
}]
xlm_max_vocab = {
  "pt": 66554,
  "it": 67802,
  "ar": 49871,
  "fr": 85704,
  "es": 87080,
  "de": 91696
}
xlm = [{
    "Model": "XLM-R\textsubscript{BASE}",
    "Language": language_map[l],
    "Vocab": pretty(xlm_max_vocab[l]/10**3)+"K",
    "Param": pretty(param_size_trimmed_xlm[xlm_max_vocab[l]]["full"]/10**6)+"M"
} for l in xlm_max_vocab] + [{
    "Model": "XLM-R\textsubscript{BASE}",
    "Language": "No Trim",
    "Vocab": pretty(param_size_full_xlm["vocab_size"]/10**3)+"K",
    "Param": pretty(param_size_full_xlm["full"]/10**6)+"M"
}]

df_1 = pd.DataFrame(mt5 + xlm)
print("- Table 1")
print(df_1.to_latex(index=False, escape=False))

output_tmp = []
for s in [15000, 30000, 45000, 60000, 75000, 90000, 105000, 120000, None]:
    if s is None:
        output_tmp.append({
            "Vocab": "250K",
            "Param (MT5)": pretty(param_size_full_mt5['full'] / 10 ** 6) + "M",
            "Param (XLM-R)": pretty(param_size_full_xlm['full'] / 10 ** 6) + "M"
        })
    else:
        output_tmp.append({
            "Vocab": pretty(s/10**3)+"K",
            "Param (MT5)": pretty(param_size_trimmed_mt5[s]['full']/10**6)+"M",
            "Param (XLM-R)": pretty(param_size_trimmed_xlm[s]['full']/10**6)+"M" if s in param_size_trimmed_xlm else "-"
        })
df_2 = pd.DataFrame(output_tmp)
print("- Table 2")
print(df_2.to_latex(index=False, escape=False))
