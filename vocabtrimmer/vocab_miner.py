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


def update_fq(tokens, fq):
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
        cache_file_frequency: str = None):
    logging.info(f"[DATASET INFO] dataset: {dataset}, name: {dataset_name}, split: {dataset_split}, column: {dataset_column}")
    logging.info(f"[MINING INFO] language: {language}, model: {model}, chunk: {chunk}")
    if not os.path.exists(cache_file_frequency):
        os.makedirs(os.path.dirname(cache_file_frequency), exist_ok=True)

        # processing dataset
        dataset = load_dataset(dataset, dataset_name, split=dataset_split)

        # tokenization
        logging.info(f"caching all tokens to {cache_file_frequency}")
        tokenizer = AutoTokenizer.from_pretrained(model)
        batch = []
        fq = defaultdict(int)
        for t in tqdm(dataset):
            batch.append(t[dataset_column])
            if len(batch) >= chunk:
                fq = update_fq(chain(*tokenizer(batch)['input_ids']), fq)
                batch = []
        if len(batch) != 0:
            fq = update_fq(chain(*tokenizer(batch)['input_ids']), fq)
        with open(cache_file_frequency, "w") as f:
            json.dump(fq, f)
        
        # with open(cache_file_frequency) as f:
        #     tokens = f.read().split("\n")
        # 
        # # count stats
        # logging.info(f"count tokens")
        # unique_token, cnt = np.unique(tokens, return_counts=True)
        # logging.info(f"caching token frequency to {cache_file_frequency}")
        # freq = dict(zip(unique_token.tolist(), cnt.tolist()))
        # with open(cache_file_frequency, "w") as f:
        #     json.dump(freq, f)

    with open(cache_file_frequency) as f:
        freq = json.load(f)


