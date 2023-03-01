import json
import logging
import os
from tqdm import tqdm
from itertools import chain
import numpy as np
from transformers import AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
from . import character_detector
from .base_trimmer import get_cache_dir


def get_token_freq(tokens):
    fq = defaultdict(int)
    for w in tokens:
        fq[w] += 1
    return fq


def vocab_miner(
        model: str = 'google/mt5-small',
        language: str = 'ja',
        dataset: str = 'mc4',
        dataset_column: str = 'text',
        dataset_split: str = 'train',
        dataset_name: str = "ja",
        chunk: int = 1000,
        cache_file_vocab: str = None,
        cache_file_frequency: str = None):

    if cache_file_frequency is None:
        cache_file_frequency = f"{get_cache_dir('cache')}/frequency.json"
    if not os.path.exists(cache_file_frequency):
        os.makedirs(os.path.dirname(cache_file_frequency), exist_ok=True)
        # processing dataset
        dataset = load_dataset(dataset, dataset_name, split=dataset_split)
        logging.info(f'{len(dataset)} texts loaded from {dataset} ({dataset_name})')

        # tokenization
        if cache_file_vocab is None:
            cache_file_vocab = f"{get_cache_dir('cache')}/vocab.json"
        if not os.path.exists(cache_file_vocab):
            os.makedirs(os.path.dirname(cache_file_vocab), exist_ok=True)
            logging.info(f"caching all tokens to {cache_file_vocab}")
            tokenizer = AutoTokenizer.from_pretrained(model)
            f_writer = open(cache_file_vocab, "w")
            batch = []
            for t in tqdm(dataset):
                batch.append(t[dataset_column])
                if len(batch) >= chunk:
                    f_writer.write(json.dumps(get_token_freq(chain(*tokenizer(batch)['input_ids']))) + "\n")
                    batch = []
            if len(batch) != 0:
                f_writer.write(json.dumps(get_token_freq(chain(*tokenizer(batch)['input_ids']))) + "\n")
            f_writer.close()
        with open(cache_file_vocab) as f:
            tokens = f.read().split("\n")

        # count stats
        logging.info(f"count tokens")
        unique_token, cnt = np.unique(tokens, return_counts=True)
        logging.info(f"caching token frequency to {cache_file_frequency}")
        freq = dict(zip(unique_token.tolist(), cnt.tolist()))
        with open(cache_file_frequency, "w") as f:
            json.dump(freq, f)

    with open(cache_file_frequency) as f:
        freq = json.load(f)


