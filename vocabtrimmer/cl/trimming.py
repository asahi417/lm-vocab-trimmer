import argparse
import logging

import vocabtrimmer
from transformers import AutoConfig


logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')


def main():
    parser = argparse.ArgumentParser(description='Trimming LM vocabulary')
    parser.add_argument('-m', '--model', required=True, type=str)
    parser.add_argument('-t', '--model-type', help='mt5/mbart/xlm-roberta', default=None, type=str)
    parser.add_argument('-l', '--language', help='language code of tokens to keep', required=True, type=str)
    parser.add_argument('-o', '--output-dir', help='directly to save model', default=None, type=str)
    parser.add_argument('--repo-id', default=None, type=str)
    parser.add_argument('--cache-dir', default='cache', type=str)
    opt = parser.parse_args()

    # check model type and load the trimmer instance
    if opt.model_type is None:
        config = AutoConfig.from_pretrained(opt.model)
        opt.model_type = config.model_type

    if opt.model_type in ['mt5']:
        trimmer = vocabtrimmer.MT5VocabTrimmer(opt.model)
    elif opt.model_type in ['mbart']:
        trimmer = vocabtrimmer.MBartVocabTrimmer(opt.model)
    else:
        trimmer = vocabtrimmer.XLMRobertaVocabTrimmer(opt.model)

    # trimming
    trimmer.trim_vocab(language=opt.language)

    # save
    if opt.repo_id is not None:  # push to huggingface
        logging.info(f"pushing to {opt.repo_id}")
        trimmer.push_to_hub(opt.repo_id)
    elif opt.output_dir is not None:  # save model to local
        logging.info(f"saving at {opt.output_dir}")
        trimmer.save_pretrained(opt.output_dir)
    trimmer.clean_cache()