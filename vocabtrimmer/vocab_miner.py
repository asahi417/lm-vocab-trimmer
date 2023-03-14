""" Mining language specific vocabulary from corpus """
import json
import logging
import os
from itertools import chain
from tqdm import tqdm

from transformers import AutoTokenizer
from datasets import load_dataset
from collections import defaultdict

from .util import DEFAULT_CACHE_DIR

__all__ = "vocab_miner"


def update_fq(tokens, fq):
    for w in tokens:
        fq[w] += 1
    return fq


def vocab_miner(model: str = 'google/mt5-small', language: str = 'ja', dataset: str = 'mc4',
                dataset_column: str = 'text', dataset_name: str = None, dataset_split: str = 'train',
                target_vocab_size: int = None, min_frequency: int = 2, chunk: int = 1000,
                cache_file_vocab: str = None, cache_file_frequency: str = None,
                overwrite: bool = False):
    """ Mining language specific vocabulary from corpus

    :param model: model name on huggingface or path to local model
    :param language: language code of tokens to keep
    :param dataset: huggingface dataset to be used for mining
    :param dataset_column: column of the dataset containing text for mining
    :param dataset_name: name of the dataset
    :param dataset_split: split of the dataset
    :param target_vocab_size: vocab size after mining
    :param min_frequency: min frequency of tokens
    :param chunk: chunk size at mining
    :param cache_file_vocab: cache directly to save the mined vocab
    :param cache_file_frequency: cache directly to save the frequency over the corpus used for vocab mining
    :return: a dictionary of {token: token_id}
    """

    dataset_name = language if dataset in ['mc4', 'vocabtrimmer/mc4_validation'] and dataset_name is None else dataset_name
    logging.info(f"[DATASET INFO] dataset: {dataset}, name: {dataset_name}, split: {dataset_split}, column: {dataset_column}")
    logging.info(f"[MINING INFO] language: {language}, model: {model}, chunk: {chunk}")

    if cache_file_frequency is None:
        cache_file_frequency = f"{DEFAULT_CACHE_DIR}/vocab_mining/frequency.{dataset}.{dataset_column}.{dataset_name}.{dataset_split}.{model.replace('/', '_')}.json"
    if cache_file_vocab is None:
        cache_file_vocab = f"{DEFAULT_CACHE_DIR}/vocab_mining/vocab.{dataset}.{dataset_column}.{dataset_name}.{dataset_split}.{model.replace('/', '_')}.{target_vocab_size}.{min_frequency}.json"

    if not overwrite and os.path.exists(cache_file_vocab):
        logging.info(f"load vocab file from {cache_file_vocab}")
        with open(cache_file_vocab) as f:
            return json.load(f)

    os.makedirs(os.path.dirname(cache_file_frequency), exist_ok=True)
    os.makedirs(os.path.dirname(cache_file_vocab), exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model)
    if not os.path.exists(cache_file_frequency):
        os.makedirs(os.path.dirname(cache_file_frequency), exist_ok=True)

        # processing dataset
        dataset = load_dataset(dataset, dataset_name, split=dataset_split)

        # tokenization
        logging.info(f"caching all tokens to {cache_file_frequency}")
        batch = []
        fq = defaultdict(int)
        for t in tqdm(dataset):
            batch.append(t[dataset_column])
            if len(batch) >= chunk:
                fq = update_fq(chain(*tokenizer(batch)['input_ids']), fq)
                batch = []
        if len(batch) != 0:
            fq = update_fq(chain(*tokenizer(batch)['input_ids']), fq)
        logging.info(f"saving frequency file to {cache_file_frequency}")
        with open(cache_file_frequency, "w") as f:
            json.dump(fq, f)

    logging.info(f"load frequency file from {cache_file_frequency}")
    with open(cache_file_frequency) as f:
        freq = [(tokenizer.convert_ids_to_tokens(int(k)), v, int(k)) for k, v in json.load(f).items() if v >= min_frequency]

    freq = sorted(freq, key=lambda x: x[1], reverse=True)
    if target_vocab_size is not None:
        assert target_vocab_size < len(freq), "vocabulary size is already smaller than the target_vocab_size"
        freq = freq[:target_vocab_size]
    new_vocab = {x[0]: x[2] for x in freq}
    logging.info(f"save vocab file at {cache_file_vocab}")
    with open(cache_file_vocab, 'w') as f:
        json.dump(new_vocab, f)
    return new_vocab




