""" Fine-tune XLM on tweet-sentiment-multilingual dataset https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual


export RAY_RESULTS='ray_results'

python lm_finetuning.py -m "roberta-large" -o "ckpt/2021/roberta-large" --push-to-hub --hf-organization "cardiffnlp" -a "roberta-large-tweet-topic-single-all" --split-train "train_all" --split-valid "validation_2021" --split-test "test_2021"
python lm_finetuning.py -m "roberta-large" -o "ckpt/2020/roberta-large" --push-to-hub --hf-organization "cardiffnlp" -a "roberta-large-tweet-topic-single-2020" --split-train "train_2020" --split-valid "validation_2020" --split-test "test_2021"
"""

import argparse
import json
import logging
import os
import shutil
import urllib.request
import multiprocessing
from os.path import join as pj

import ray
import torch
import numpy as np
from huggingface_hub import create_repo
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from ray import tune

from readme import get_readme

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')

os.environ["WANDB_DISABLED"] = "true"
PARALLEL = bool(int(os.getenv("PARALLEL", 1)))
RAY_RESULTS = os.getenv("RAY_RESULTS", "ray_results")
LABEL2ID = {
        "arts_&_culture": 0,
        "business_&_entrepreneurs": 1,
        "pop_culture": 2,
        "daily_life": 3,
        "sports_&_gaming": 4,
        "science_&_technology": 5
    }
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def internet_connection(host='http://google.com'):
    try:
        urllib.request.urlopen(host)
        return True
    except:
        return False


def get_metrics():
    metric_accuracy = load_metric("accuracy")
    metric_f1 = load_metric("f1")

    def compute_metric_search(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric_f1.compute(predictions=predictions, references=labels, average='micro')

    def compute_metric_all(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'f1': metric_f1.compute(predictions=predictions, references=labels, average='micro')['f1'],
            'f1_macro': metric_f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
            'accuracy': metric_accuracy.compute(predictions=predictions, references=labels)['accuracy']
        }
    return compute_metric_search, compute_metric_all


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning language model.')
    parser.add_argument('-m', '--model', help='transformer LM', default='roberta-base', type=str)
    parser.add_argument('-d', '--dataset', help='', default='cardiffnlp/tweet_topic_single', type=str)
    parser.add_argument('--split-train', help='', required=True, type=str)
    parser.add_argument('--split-validation', help='', required=True, type=str)
    parser.add_argument('--split-test', help='', required=True, type=str)
    parser.add_argument('-l', '--seq-length', help='', default=128, type=int)
    parser.add_argument('--random-seed', help='', default=42, type=int)
    parser.add_argument('--eval-step', help='', default=50, type=int)
    parser.add_argument('-o', '--output-dir', help='Directory to output', default='ckpt_tmp', type=str)
    parser.add_argument('-t', '--n-trials', default=10, type=int)
    parser.add_argument('--num-cpus', default=1, type=int)
    parser.add_argument('--push-to-hub', action='store_true')
    parser.add_argument('--use-auth-token', action='store_true')
    parser.add_argument('--hf-organization', default=None, type=str)
    parser.add_argument('-a', '--model-alias', help='', default=None, type=str)
    parser.add_argument('--summary-file', default='metric_summary.json', type=str)
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    opt = parser.parse_args()
    assert opt.summary_file.endswith('.json'), f'`--summary-file` should be a json file {opt.summary_file}'

    ray.init(ignore_reinit_error=True, num_cpus=opt.num_cpus)

    # setup data
    dataset = load_dataset(opt.dataset)
    network = internet_connection()
    # setup model
    tokenizer = AutoTokenizer.from_pretrained(opt.model, local_files_only=not network)
    model = AutoModelForSequenceClassification.from_pretrained(
        opt.model,
        num_labels=6,
        local_files_only=not network,
        id2label=ID2LABEL,
        label2id=LABEL2ID
        )
    tokenized_datasets = dataset.map(
        lambda x: tokenizer(x["text"], padding="max_length", truncation=True, max_length=opt.seq_length),
        batched=True)
    # setup metrics
    compute_metric_search, compute_metric_all = get_metrics()

    if not opt.skip_train:
        # setup trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=opt.output_dir,
                evaluation_strategy="steps",
                eval_steps=opt.eval_step,
                seed=opt.random_seed
            ),
            train_dataset=tokenized_datasets[opt.split_train],
            eval_dataset=tokenized_datasets[opt.split_validation],
            compute_metrics=compute_metric_search,
            model_init=lambda x: AutoModelForSequenceClassification.from_pretrained(
                opt.model,
                num_labels=6,
                local_files_only=not network,
                return_dict=True,
                id2label=ID2LABEL,
                label2id=LABEL2ID
            )
        )
        # parameter search
        if PARALLEL:
            best_run = trainer.hyperparameter_search(
                hp_space=lambda x: {
                    "learning_rate": tune.loguniform(1e-6, 1e-4),
                    "num_train_epochs": tune.choice(list(range(1, 6))),
                    "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
                },
                local_dir=RAY_RESULTS, direction="maximize", backend="ray", n_trials=opt.n_trials,
                resources_per_trial={'cpu': multiprocessing.cpu_count(), "gpu": torch.cuda.device_count()},

            )
        else:
            best_run = trainer.hyperparameter_search(
                hp_space=lambda x: {
                    "learning_rate": tune.loguniform(1e-6, 1e-4),
                    "num_train_epochs": tune.choice(list(range(1, 6))),
                    "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
                },
                local_dir=RAY_RESULTS, direction="maximize", backend="ray", n_trials=opt.n_trials
            )
        # finetuning
        for n, v in best_run.hyperparameters.items():
            setattr(trainer.args, n, v)
        trainer.train()
        trainer.save_model(pj(opt.output_dir, 'best_model'))
        best_model_path = pj(opt.output_dir, 'best_model')
    else:
        best_model_path = pj(opt.output_dir, 'best_model')

    # evaluation
    model = AutoModelForSequenceClassification.from_pretrained(
        best_model_path,
        num_labels=6,
        local_files_only=not network,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=opt.output_dir,
            evaluation_strategy="no",
            seed=opt.random_seed
        ),
        train_dataset=tokenized_datasets[opt.split_train],
        eval_dataset=tokenized_datasets[opt.split_test],
        compute_metrics=compute_metric_all,
    )
    summary_file = pj(opt.output_dir, opt.summary_file)
    if not opt.skip_eval:
        result = {f'test/{k}': v for k, v in trainer.evaluate().items()}
        logging.info(json.dumps(result, indent=4))
        with open(summary_file, 'w') as f:
            json.dump(result, f)

    if opt.push_to_hub:
        assert opt.hf_organization is not None, f'specify hf organization `--hf-organization`'
        assert opt.model_alias is not None, f'specify hf organization `--model-alias`'
        url = create_repo(opt.model_alias, organization=opt.hf_organization, exist_ok=True)
        # if not opt.skip_train:
        args = {"use_auth_token": opt.use_auth_token, "repo_url": url, "organization": opt.hf_organization}
        trainer.model.push_to_hub(opt.model_alias, **args)
        tokenizer.push_to_hub(opt.model_alias, **args)
        if os.path.exists(summary_file):
            shutil.copy2(summary_file, opt.model_alias)
        extra_desc = f"This model is fine-tuned on `{opt.split_train}` split and validated on `{opt.split_test}` split of tweet_topic."
        readme = get_readme(
            model_name=f"{opt.hf_organization}/{opt.model_alias}",
            metric=summary_file,
            language_model=opt.model,
            extra_desc= extra_desc
            )
        with open(f"{opt.model_alias}/README.md", "w") as f:
            f.write(readme)
        os.system(
            f"cd {opt.model_alias} && git lfs install && git add . && git commit -m 'model update' && git push && cd ../")
        shutil.rmtree(f"{opt.model_alias}")  # clean up the cloned repo


if __name__ == '__main__':
    main()
