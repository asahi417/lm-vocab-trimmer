import logging
import json
import os
from vocabtrimmer import MT5VocabTrimmer

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

logging.info("Trimming QG models")
export_dir = 'model_qg'
language = ['ja', 'ko', 'ru', 'fr', 'de', 'es', 'it']
stats = []
if os.path.exists(f"{export_dir}/stats.jsonl"):
    with open(f"{export_dir}/stats.jsonl", 'r') as f:
        stats += [json.loads(line) for line in f.read().split('\n') if len(line) > 0]

for size in ['small', 'base']:
    for la in language:
        model_ckpt = f'{export_dir}/mt5-{size}-{la}quad-qg-trimmed'
        if os.path.exists(model_ckpt):
            continue

        logging.info(f"Language: {la}, Size: {la}, Model: lmqg/mt5-{size}-{la}quad-qg")

        _stats_tmp = {"model": f'lmqg/mt5-{size}-{la}quad-qg'}
        trimmer = MT5VocabTrimmer(f'lmqg/mt5-{size}-{la}quad-qg')
        _stats_tmp["original/model_size_full"] = trimmer.model_size_full
        _stats_tmp["original/model_size_embedding"] = trimmer.model_size_embedding
        _stats_tmp["original/model_size_rest"] = trimmer.model_size_full - trimmer.model_size_embedding

        trimmer.trim_vocab(language=la)
        _stats_tmp["trimmed/model_size_full"] = trimmer.model_size_full
        _stats_tmp["trimmed/model_size_embedding"] = trimmer.model_size_embedding
        _stats_tmp["trimmed/model_size_rest"] = trimmer.model_size_full - trimmer.model_size_embedding

        trimmer.save_pretrained(model_ckpt)
        logging.info(f'model saved at `{model_ckpt}`')
        stats.append(_stats_tmp)

        with open(f"{export_dir}/stats.jsonl", "w") as f:
            f.write('\n'.join([json.dumps(s) for s in stats]))
