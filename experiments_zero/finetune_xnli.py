""" Fine-tune LM on multilabel classification task. """

import argparse
import json
import logging
import os
import shutil
from os.path import join as pj

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from huggingface_hub import Repository
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
os.environ["WANDB_DISABLED"] = "true"


def main():
    parser = argparse.ArgumentParser(description='Fine-tuning language model.')
    parser.add_argument('-o', '--output-dir', help='Directory to output', required=True, type=str)
    parser.add_argument('-m', '--model', help='transformer LM', default='roberta-base', type=str)
    parser.add_argument('-d', '--dataset', help='', default='xnli', type=str)
    parser.add_argument('-n', '--dataset-name', help='', default='en', type=str)
    parser.add_argument('-a', '--attn-implementation', help='', default='sdpa', type=str)
    parser.add_argument('--column-premise', help='', default='premise', type=str)
    parser.add_argument('--column-hypothesis', help='', default='hypothesis', type=str)
    parser.add_argument('--column-output', help='', default='label', type=str)
    parser.add_argument('--split-train', help='', default='train', type=str)
    parser.add_argument('--split-validation', help='', default='validation', type=str)
    parser.add_argument('--split-test', help='', default='test', type=str)
    parser.add_argument('-l', '--seq-length', help='', default=256, type=int)
    parser.add_argument('--random-seed', help='', default=42, type=int)
    parser.add_argument('--eval-step', help='', default=500, type=int)
    parser.add_argument('--repo-id', default=None, type=str)
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--skip-eval', action='store_true')
    parser.add_argument('--lr', help='', default=0.000001, type=float)
    parser.add_argument('-b', '--batch', help='', default=256, type=int)
    parser.add_argument('-e', '--epoch', help='', default=5, type=int)
    opt = parser.parse_args()

    # setup data
    dataset = load_dataset(opt.dataset, opt.dataset_name)
    id2label = {n: k for n, k in enumerate(dataset[opt.split_train].features[opt.column_output].names)}
    label2id = {v: k for k, v in id2label.items()}
    
    # setup model
    tokenizer = AutoTokenizer.from_pretrained(opt.model)
    tokenized_datasets = dataset.map(
        lambda x: tokenizer(
            x[opt.column_premise],
            x[opt.column_hypothesis],
            padding="max_length",
            truncation=True,
            max_length=opt.seq_length
        ),
        batched=True
    )

    # setup metrics
    metric_accuracy = load("accuracy")
    metric_f1 = load("f1")
    metric_pre = load("precision")
    metric_re = load("recall")

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
    best_model_path = pj(opt.output_dir, 'best_model')
    metric_file = pj(opt.output_dir, "eval.json")
    model_config = dict(
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
        attn_implementation=opt.attn_implementation,
        torch_dtype=torch.bfloat16
    )
    if not opt.skip_train:
        trainer = Trainer(
            model=AutoModelForSequenceClassification.from_pretrained(opt.model, **model_config),
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
            args=TrainingArguments(
                output_dir=opt.output_dir,
                evaluation_strategy="steps",
                eval_steps=opt.eval_step,
                save_steps=opt.eval_step,
                load_best_model_at_end=True,
                learning_rate=opt.lr,
                num_train_epochs=opt.epoch,
                per_device_train_batch_size=opt.batch,
                seed=opt.random_seed),
            train_dataset=tokenized_datasets[opt.split_train],
            eval_dataset=tokenized_datasets[opt.split_validation],
            compute_metrics=compute_metric_search
        )
        trainer.train()
        trainer.save_model(best_model_path)
    if not opt.skip_eval:
        eval_config = dict(
            model=AutoModelForSequenceClassification.from_pretrained(best_model_path, **model_config),
            args=TrainingArguments(output_dir=opt.output_dir, evaluation_strategy="no", seed=opt.random_seed),
            train_dataset=tokenized_datasets[opt.split_train],
            compute_metrics=compute_metric_all
        )
        metric_test = Trainer(**eval_config, eval_dataset=tokenized_datasets[opt.split_test]).evaluate()
        metric_valid = Trainer(**eval_config, eval_dataset=tokenized_datasets[opt.split_validation]).evaluate()
        metric = {"test": metric_test, "validation": metric_valid}
        logging.info(json.dumps(metric, indent=4))
        with open(metric_file, 'w') as f:
            json.dump(metric, f)
    if opt.repo_id is not None:
        model = AutoModelForSequenceClassification.from_pretrained(best_model_path, **model_config)
        model.push_to_hub(opt.repo_id)
        tokenizer.push_to_hub(opt.repo_id)
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
