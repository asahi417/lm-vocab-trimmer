""" Fine-tune LM on multilabel classification task. """

import argparse
import json
import logging
import os
import shutil

import multiprocessing
from os.path import join as pj

import ray
import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from huggingface_hub import Repository
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from ray import tune

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
os.environ["WANDB_DISABLED"] = "true"


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning language model.')
    parser.add_argument('-o', '--output-dir', help='Directory to output', required=True, type=str)
    parser.add_argument('-m', '--model', help='transformer LM', default='xlm-roberta-base', type=str)
    parser.add_argument('-d', '--dataset', help='', default='cardiffnlp/tweet_sentiment_multilingual', type=str)
    parser.add_argument('-n', '--dataset-name', help='', default='english', type=str)
    parser.add_argument('--column-input', help='', default='text', type=str)
    parser.add_argument('--column-output', help='', default='label', type=str)
    parser.add_argument('--split-train', help='', default='train', type=str)
    parser.add_argument('--split-validation', help='', default='validation', type=str)
    parser.add_argument('--split-test', help='', default='test', type=str)
    parser.add_argument('-r', '--ray-result-dir', help='', default='ray_result', type=str)
    parser.add_argument('-l', '--seq-length', help='', default=128, type=int)
    parser.add_argument('--random-seed', help='', default=42, type=int)
    parser.add_argument('--eval-step', help='', default=50, type=int)
    parser.add_argument('-t', '--n-trials', default=10, type=int)
    parser.add_argument('--num-cpus', default=1, type=int)
    parser.add_argument('--repo-id', default=None, type=str)
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--parallel', action='store_true')
    opt = parser.parse_args()

    ray.init(ignore_reinit_error=True, num_cpus=opt.num_cpus)

    # setup data
    dataset = load_dataset(opt.dataset, opt.dataset_name)
    id2label = {n: k for n, k in enumerate(dataset[opt.split_train].features[opt.column_output].names)}
    label2id = {v: k for k, v in id2label.items()}
    
    # setup model
    tokenizer = AutoTokenizer.from_pretrained(opt.model)
    model = AutoModelForSequenceClassification.from_pretrained(
        opt.model, num_labels=len(label2id), id2label=id2label, label2id=label2id)
    tokenized_datasets = dataset.map(
        lambda x: tokenizer(x[opt.column_input], padding="max_length", truncation=True, max_length=opt.seq_length),
        batched=True)
    
    # setup metrics
    metric_accuracy = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    metric_pre = load_metric("precision")
    metric_re = load_metric("recall")

    def compute_metric_search(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric_f1.compute(predictions=predictions, references=labels, average='micro')

    def compute_metric_all(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {
            'f1_micro': metric_f1.compute(predictions=predictions, references=labels, average='micro')['f1'],
            'recall_micro': metric_re.compute(predictions=predictions, references=labels, average='micro')['recall'],
            'precision_micro': metric_pre.compute(predictions=predictions, references=labels, average='micro')['precision'],
            'f1_macro': metric_f1.compute(predictions=predictions, references=labels, average='macro')['f1'],
            'recall_macro': metric_re.compute(predictions=predictions, references=labels, average='macro')['recall'],
            'precision_macro': metric_pre.compute(predictions=predictions, references=labels, average='macro')['precision'],
            'accuracy': metric_accuracy.compute(predictions=predictions, references=labels)['accuracy']
        }
        
    if not opt.skip_train:

        # setup trainer
        trainer = Trainer(
            model=model,
            args=TrainingArguments(
                output_dir=opt.output_dir, evaluation_strategy="steps", eval_steps=opt.eval_step, seed=opt.random_seed),
            train_dataset=tokenized_datasets[opt.split_train],
            eval_dataset=tokenized_datasets[opt.split_validation],
            compute_metrics=compute_metric_search,
            model_init=lambda x: AutoModelForSequenceClassification.from_pretrained(
                opt.model, num_labels=len(id2label), return_dict=True, id2label=id2label, label2id=label2id))

        # parameter search
        if opt.parallel:
            best_run = trainer.hyperparameter_search(
                hp_space=lambda x: {
                    "learning_rate": tune.loguniform(1e-6, 1e-4),
                    "num_train_epochs": tune.choice(list(range(1, 6))),
                    "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
                },
                local_dir=opt.ray_result_dir,
                direction="maximize",
                backend="ray",
                n_trials=opt.n_trials,
                resources_per_trial={'cpu': multiprocessing.cpu_count(), "gpu": torch.cuda.device_count()})
        else:
            best_run = trainer.hyperparameter_search(
                hp_space=lambda x: {
                    "learning_rate": tune.loguniform(1e-6, 1e-4),
                    "num_train_epochs": tune.choice(list(range(1, 6))),
                    "per_device_train_batch_size": tune.choice([4, 8, 16, 32, 64]),
                },
                local_dir=opt.ray_result_dir,
                direction="maximize",
                backend="ray",
                n_trials=opt.n_trials)

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
        best_model_path, num_labels=len(id2label), id2label=id2label, label2id=label2id)
    trainer = Trainer(
        model=model,
        args=TrainingArguments(output_dir=opt.output_dir, evaluation_strategy="no", seed=opt.random_seed),
        train_dataset=tokenized_datasets[opt.split_train],
        eval_dataset=tokenized_datasets[opt.split_test],
        compute_metrics=compute_metric_all)

    metric_file = pj(opt.output_dir, "eval.json")
    if not opt.skip_eval:
        metric = trainer.evaluate()
        logging.info(json.dumps(metric, indent=4))
        with open(metric_file, 'w') as f:
            json.dump(metric, f)

    if opt.repo_id is not None:

        # push model/tokenizer to hub
        trainer.model.push_to_hub(opt.repo_id)
        tokenizer.push_to_hub(opt.repo_id)

        # push metric and readme files
        repo = Repository(os.path.basename(opt.repo_id), opt.repo_id)
        df = None
        if os.path.exists(metric_file):
            shutil.copy2(metric_file, os.path.basename(opt.repo_id))
            with open(metric_file) as f:
                metric = json.load(f)
            df = pd.DataFrame([metric])[["eval_f1_micro", 'eval_recall_micro', "eval_precision_micro", "eval_f1_macro", "eval_recall_macro", "eval_precision_macro", "eval_accuracy"]]
            df = (df * 100).round(2)

        # get readme
        readme = f""" # `{opt.repo_id}`
This model is a fine-tuned version of [{opt.model}](https://huggingface.co/{opt.model}) on the 
[{opt.dataset}](https://huggingface.co/datasets/{opt.dataset}) ({opt.dataset_name}).
Following metrics are computed on the `{opt.split_test}` split of 
[{opt.dataset}](https://huggingface.co/datasets/{opt.dataset})({opt.dataset_name}). 

{df.to_markdown() if df is not None else ""}

Check the result file [here](https://huggingface.co/{opt.repo_id}/raw/main/eval.json)."""
        with open(f"{os.path.basename(opt.repo_id)}/README.md", "w") as f:
            f.write(readme)
        repo.push_to_hub()


if __name__ == '__main__':
    main()
