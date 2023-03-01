import argparse
import logging

import vocabtrimmer


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Collecting lagnuage-specific vocabulary on dataset')
    parser.add_argument('-m', '--model', default='google/mt5-small', type=str)
    parser.add_argument('-c', '--chunk-size', default=1000, type=int)
    parser.add_argument('-d', '--dataset', default='mc4', type=str)
    parser.add_argument('-n', '--dataset-name', default='ja', type=str)
    parser.add_argument('-s', '--dataset-split', default='validation', type=str)
    parser.add_argument('-l', '--language', help='language code of tokens to keep', default="ja", type=str)
    parser.add_argument('-o', '--output-dir', help='directly to save model', default='data/ja', type=str)
    parser.add_argument('--dataset-column', default='text', type=str)
    opt = parser.parse_args()

    vocabtrimmer.vocab_miner(
        model=opt.model,
        language=opt.language,
        dataset=opt.dataset,
        dataset_column=opt.dataset_column,
        dataset_split=opt.dataset_split,
        dataset_name=opt.dataset_name,
        chunk=opt.chunk_size,
        cache_file_frequency=f"{opt.output_dir}/frequency.json"
    )
