""" Mining language specific vocabulary from corpus """
import json
import logging
import os
import string
import re
import unicodedata as ud
from itertools import chain
from tqdm import tqdm

from transformers import AutoTokenizer
from datasets import load_dataset
from collections import defaultdict
from .util import pretty, DEFAULT_CACHE_DIR

__all__ = "vocab_miner"
character_greek = ''.join([chr(c) for c in list(chain(range(0x370, 0x3e2), range(0x3f0, 0x400)))])
currency = "".join([
    '؋', 'L', '֏', 'ƒ', '$', '$', 'ƒ', '₼', '$', '৳', '$', '$', '$', '฿', 'P', '$', '$', '¥', '$', '₡', '$',
    '₱', '$', '£', 'Ξ', '€', '$', '£', '£', '₾', '£', '₵', '£', 'D', 'Q', '$', '$', 'L', 'G', '₪', '£', '₹',
    '﷼', '£', '¥', '៛', '₩', '₩', '$', '₸', '₭', '£', '₨', '$', 'M', 'Ł', 'K', '₮', '₨', '$', '$', '₦', '₨',
    '$', '﷼', 'K', '₱', '₨', '﷼', '￥', '₽', '﷼', '$', '₨', '$', '£', 'S', '$', '£', '$', '£', 'E', '฿', 'T',
    '₤', '₺', '$', '₴', '$', '₫', 'Ƀ', '$', '₣', '﷼', 'R'])
digit = '0123456789'
symbol = string.punctuation + '()¬”§¤ˆ⁄¶“士•⟨⟩∫ˈΔ†′‐−—▁⁄–▁́°×»’«≈‰∞⇨‡→・±’”→〜】יחַיים√′→‚’“ʾ“„’‘”≈ʿ' + character_greek + "".join(currency)
symbol_es = '¿¡'
symbol_ja = "〈〉「」『』《》。、！？"
range_ja = [
        {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},  # compatibility ideographs
        {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},  # compatibility ideographs
        {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},  # compatibility ideographs
        {"from": ord(u"\U0002F800"), "to": ord(u"\U0002fa1f")},  # compatibility ideographs
        {'from': ord(u'\u3040'), 'to': ord(u'\u309f')},  # Japanese Hiragana
        {"from": ord(u"\u30a0"), "to": ord(u"\u30ff")},  # Japanese Katakana
        {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},  # cjk radicals supplement
        {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
        {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
        {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
        {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
        {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
        {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}  # included as of Unicode 8.0
    ]
range_zh = [
        {"from": ord(u"\u3300"), "to": ord(u"\u33ff")},  # compatibility ideographs
        {"from": ord(u"\ufe30"), "to": ord(u"\ufe4f")},  # compatibility ideographs
        {"from": ord(u"\uf900"), "to": ord(u"\ufaff")},  # compatibility ideographs
        {"from": ord(u"\U0002F800"), "to": ord(u"\U0002fa1f")},  # compatibility ideographs
        {"from": ord(u"\u2e80"), "to": ord(u"\u2eff")},  # cjk radicals supplement
        {"from": ord(u"\u4e00"), "to": ord(u"\u9fff")},
        {"from": ord(u"\u3400"), "to": ord(u"\u4dbf")},
        {"from": ord(u"\U00020000"), "to": ord(u"\U0002a6df")},
        {"from": ord(u"\U0002a700"), "to": ord(u"\U0002b73f")},
        {"from": ord(u"\U0002b740"), "to": ord(u"\U0002b81f")},
        {"from": ord(u"\U0002b820"), "to": ord(u"\U0002ceaf")}  # included as of Unicode 8.0
    ]
hangul_ranges = [
        range(0xAC00, 0xD7A4),  # Hangul Syllables (AC00–D7A3)
        range(0x1100, 0x1200),  # Hangul Jamo (1100–11FF)
        range(0x3130, 0x3190),  # Hangul Compatibility Jamo (3130-318F)
        range(0xA960, 0xA980),  # Hangul Jamo Extended-A (A960-A97F)
        range(0xD7B0, 0xD800),  # Hangul Jamo Extended-B (D7B0-D7FF)
    ]


def norm(term: str): return ud.normalize('NFKC', term.lower())


def is_latin(term: str):
    latin_letters = {}

    def _is_latin(uchr):
        try:
            return latin_letters[uchr]
        except KeyError:
            return latin_letters.setdefault(uchr, 'LATIN' in ud.name(uchr))

    return any(_is_latin(c) for c in norm(term) if c.isalpha())


def is_en(term: str): return bool(re.search(rf"[{string.ascii_lowercase}]", norm(term)))


def is_es(term: str): return is_latin(term) or bool(re.search(rf"[{symbol_es}]", norm(term)))


def is_ar(term: str): return bool(re.search(r"[\u0621-\u064a\ufb50-\ufdff\ufe70-\ufefc]", norm(term)))


def is_ru(term: str): return bool(re.search('[а-яА-Я]', norm(term)))


def is_ja(term: str): return bool(re.search(f"[{symbol_ja}]", norm(term))) or any(any([r["from"] <= ord(c) <= r["to"] for r in range_ja]) for c in norm(term))


def is_zh(term: str): return any(any([r["from"] <= ord(c) <= r["to"] for r in range_zh]) for c in norm(term))


def is_ko(term: str): return any(any(ord(c) in r for r in hangul_ranges) for c in norm(term))


def is_language(term: str, language: str):
    if language.lower() == 'en':
        return is_en(term)
    elif language.lower() == 'es':
        return is_es(term)
    elif language.lower() == 'ar':
        return is_ar(term)
    elif language.lower() == 'ko':
        return is_ko(term)
    elif language.lower() == 'ja':
        return is_ja(term)
    elif language.lower() == 'zh':
        return is_zh(term)
    elif language.lower() == 'ru':
        return is_ru(term)
    return is_latin(term)


def update_fq(tokens, fq):
    for w in tokens:
        fq[w] += 1
    return fq


def vocab_miner(model: str = 'google/mt5-small', language: str = 'ja', dataset: str = 'mc4',
                dataset_column: str = 'text', dataset_name: str = None, dataset_split: str = 'train',
                target_vocab_size: int = None, min_frequency: int = 2, chunk: int = 1000,
                cache_file_vocab: str = None, cache_file_frequency: str = None):
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

    if os.path.exists(cache_file_vocab):
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
        flag_native = [is_language(x[0], language) for x in freq]  # flag whether the token is native language
        logging.info(f"Number of native token: {pretty(sum(flag_native))}/{pretty(len(freq))} ({round(sum(flag_native) / len(freq) * 100, 2)}%)")

    if target_vocab_size is None:
        final_tokens = freq
    elif target_vocab_size >= len(freq):
        raise ValueError("vocabulary size is already smaller than the target_vocab_size")
    elif target_vocab_size < sum(flag_native):
        logging.warning("target_vocab_size is smaller than the native language token")
        final_tokens = sorted([x for x, y in zip(freq, flag_native) if y], key=lambda x: x[1], reverse=True)[:target_vocab_size]
    else:
        n_ool_token = target_vocab_size - sum(flag_native)
        logging.info(f"keep {pretty(n_ool_token)} tokens from out-of-language tokens")
        ool_tokens = sorted([x for x, y in zip(freq, flag_native) if not y], key=lambda x: x[1], reverse=True)[:n_ool_token]
        final_tokens = [x for x, y in zip(freq, flag_native) if y] + ool_tokens
    new_vocab = {x[0]: x[2] for x in final_tokens}
    logging.info(f"save vocab file at {cache_file_vocab}")
    with open(cache_file_vocab, 'w') as f:
        json.dump(new_vocab, f)
    return new_vocab




