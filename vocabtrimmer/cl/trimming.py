import argparse
import logging

import vocabtrimmer


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Trimming LM vocabulary')

    # vocab trimming parameter
    parser.add_argument('-m', '--model', default='google/mt5-small', help="model name on huggingface or path to local model", type=str)
    parser.add_argument('-l', '--language', help='language code of tokens to keep', required=True, type=str)
    parser.add_argument('-p', '--path-to-save', help='directly to save model', required=True, type=str)
    parser.add_argument('--repo-id', help='huggingface repo id to push after trimming', default=None, type=str)

    # vocab mining parameter
    parser.add_argument('-c', '--chunk', help="chunk size at mining", default=500, type=int)
    parser.add_argument('-v', '--target-vocab-size', help="vocab size after mining", default=None, type=int)
    parser.add_argument('-d', '--dataset', help='huggingface dataset to be used for mining', default='vocabtrimmer/mc4_validation', type=str)
    parser.add_argument('-n', '--dataset-name', help='name of the dataset', default=None, type=str)
    parser.add_argument('-s', '--dataset-split', help='split of the dataset', default='validation', type=str)
    parser.add_argument('--dataset-column', help="column of the dataset containing text for mining", default='text', type=str)
    parser.add_argument('--cache-file-vocab', help="cache directly to save the mined vocab", default=None, type=str)
    parser.add_argument('--cache-file-frequency', help="cache directly to save the frequency over the corpus used for vocab mining", default=None, type=str)
    parser.add_argument('--min-frequency', help="min frequency of tokens", default=2, type=int)
    parser.add_argument('--tokens-to-keep', help='custom tokens to keep in vocabulary', nargs='+', default=None, type=str)

    parser.add_argument('--overwrite', help='', action='store_true')

    opt = parser.parse_args()

    # trimming
    trimmer = vocabtrimmer.VocabTrimmer(opt.model)
    trimmer.trim_vocab(
        path_to_save=opt.path_to_save,
        language=opt.language,
        dataset=opt.dataset,
        dataset_column=opt.dataset_column,
        dataset_name=opt.dataset_name,
        dataset_split=opt.dataset_split,
        target_vocab_size=opt.target_vocab_size,
        min_frequency=opt.min_frequency,
        chunk=opt.chunk,
        cache_file_vocab=opt.cache_file_vocab,
        cache_file_frequency=opt.cache_file_frequency,
        overwrite=opt.overwrite
    )

    # push to huggingface
    if opt.repo_id is not None:
        logging.info(f"pushing to {opt.repo_id}")
        trimmer.push_to_hub(opt.repo_id)
