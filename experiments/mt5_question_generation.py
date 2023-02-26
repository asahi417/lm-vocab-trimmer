import logging
import json
import os

from lmqg import evaluate
from vocabtrimmer import MT5VocabTrimmer

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

logging.info("Trimming QG models")
export_dir = 'model_qg'
language = ['ja', 'ko', 'ru', 'fr', 'de', 'es', 'it']
model_size = ['small', 'base']
stats = []
if os.path.exists(f"{export_dir}/stats.jsonl"):
    with open(f"{export_dir}/stats.jsonl", 'r') as f:
        stats += [json.loads(line) for line in f.read().split('\n') if len(line) > 0]

for size in model_size:
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

        trimmer.trim_vocab(language=la, path_to_save=model_ckpt)
        _stats_tmp["trimmed/model_size_full"] = trimmer.model_size_full
        _stats_tmp["trimmed/model_size_embedding"] = trimmer.model_size_embedding
        _stats_tmp["trimmed/model_size_rest"] = trimmer.model_size_full - trimmer.model_size_embedding

        stats.append(_stats_tmp)
        with open(f"{export_dir}/stats.jsonl", "w") as f:
            f.write('\n'.join([json.dumps(s) for s in stats]))


logging.info("Evaluate Trimmed QG models")
for size in model_size:

    for la in language:
        model_ckpt = f'{export_dir}/mt5-{size}-{la}quad-qg-trimmed'

        logging.info(f"Language: {la}, Size: {la}, Model: {model_ckpt}")
        metric = evaluate(
            export_dir=model_ckpt,
            batch_size=32,
            n_beams=4,
            model=model_ckpt,
            max_length=512,
            max_length_output=64,
            dataset_path=f"lmqg/qg_{la}quad",
            input_type='paragraph_answer',
            output_type='question',
            language=la,
            bleu_only=True,
        )
        logging.info(json.dumps(metric, indent=4, sort_keys=True))