"""
for LA in 'ja' 'ko' 'ru' 'fr' 'de' 'es' 'it'
do
    for SIZE in 'small' 'base'
    do
        lmqg-eval -m "model_qg/mt5-${SIZE}-${LA}quad-qg-trimmed" -e "model_qg/mt5-${SIZE}-${LA}quad-qg-trimmed" --language "${LA}" -d "lmqg/qg_${LA}quad" -i "paragraph_answer"
    done
done
lmqg-eval -m "model_qg/mt5-small-jaquad-qg-trimmed" -e "model_qg/mt5-small-jaquad-qg-trimmed" --language "ja" -d "lmqg/qg_jaquad" -i "paragraph_answer"
"""
import logging
import json
import os
import requests

import pandas as pd
from lmqg import evaluate
from vocabtrimmer import MT5VocabTrimmer


def download(filename, url):
    try:
        with open(filename) as f_reader:
            json.load(f_reader)
    except Exception:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        print(url)
        with open(filename, "wb") as f_writer:
            r = requests.get(url)
            f_writer.write(r.content)
    with open(filename) as f_reader:
        return json.load(f_reader)


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
metrics = []
for size in model_size:

    for la in language:
        model_ckpt = f'{export_dir}/mt5-{size}-{la}quad-qg-trimmed'

        if not os.path.exists(f"{model_ckpt}/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json"):
            logging.info(f"Language: {la}, Size: {la}, Model: {model_ckpt}")
            metric = evaluate(
                export_dir=model_ckpt,
                batch_size=32,
                n_beams=4,
                model=model_ckpt,
                max_length=512,
                max_length_output=64,
                prediction_aggregation="first",
                prediction_level="sentence",
                dataset_path=f"lmqg/qg_{la}quad",
                input_type='paragraph_answer',
                output_type='question',
                language=la,
            )
            logging.info(json.dumps(metric, indent=4, sort_keys=True))
        with open(f"{model_ckpt}/metric.first.sentence.paragraph_answer.question.lmqg_qg_{la}quad.default.json") as f:
            metric_trimmed = json.load(f)
        metric_raw = download(
            filename=f"cache/mt5-{size}-{la}quad",
            url=f"https://huggingface.co/lmqg/mt5-{size}-{la}quad-qg/raw/main/eval/metric.first.answer.paragraph.questions_answers.lmqg_qg_{la}quad.default.json")
        metric = {f"{k}/trimmed": v for k, v in metric_trimmed['test'].items()}
        metric.update({f"{k}/raw": v for k, v in metric_raw['test'].items()if k in metric_trimmed['test']})
        metric['model'] = model_ckpt
        metrics.append(metric)
df = pd.DataFrame(metrics)
df.index = df.pop('model')
df = df[sorted(df.columns)]

