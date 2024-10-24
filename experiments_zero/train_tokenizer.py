import argparse
from transformers import AutoTokenizer
from datasets import load_dataset

parser = argparse.ArgumentParser(description='Tokenizer training')
parser.add_argument('-b', '--batch-size', help='', default=1_000, type=int)
parser.add_argument('-r', '--reference-model', help='', default='roberta-base', type=str)
parser.add_argument('-d', '--dataset', help='', default='vocabtrimmer/mc4_validation', type=str)
parser.add_argument('--dataset-config', help='', default='zh', type=str)
parser.add_argument('--dataset-split', help='', default='validation', type=str)
parser.add_argument('--dataset-column', help='', default='text', type=str)
parser.add_argument('-o', '--output-dir', required=True, type=str)
parser.add_argument('--repo-id', default=None, type=str)
opt = parser.parse_args()


def batch_iterator(dataset):
    for i in range(0, len(dataset), opt.batch_size):
        yield dataset[i: i + opt.batch_size][opt.dataset_column]

reference_tokenizer = AutoTokenizer.from_pretrained(opt.reference_model)
new_tokenizer = reference_tokenizer.train_new_from_iterator(
    batch_iterator(load_dataset(opt.dataset, opt.dataset_config, split=opt.dataset_split)),
    vocab_size=reference_tokenizer.vocab_size,
)
new_tokenizer.save_pretrained(opt.output_dir)
if opt.repo_id:
    new_tokenizer.push_to_hub(opt.repo_id)
