""" Mining language specific vocabulary from corpus """
import json
import logging
import os
from itertools import chain
from tqdm import tqdm

from tokenizers import models
import torch
from transformers import AutoTokenizer, AutoConfig, MBartForConditionalGeneration
from datasets import load_dataset
from collections import defaultdict

DEFAULT_CACHE_DIR = f"{os.path.expanduser('~')}/.cache/vocabtrimmer"
__all__ = "vocab_miner"
MBART_LANG_ID = ['ar_AR', 'cs_CZ', 'de_DE', 'en_XX', 'es_XX', 'et_EE', 'fi_FI', 'fr_XX', 'gu_IN', 'hi_IN', 'it_IT',
                 'ja_XX', 'kk_KZ', 'ko_KR', 'lt_LT', 'lv_LV', 'my_MM', 'ne_NP', 'nl_XX', 'ro_RO', 'ru_RU', 'si_LK',
                 'tr_TR', 'vi_VN', 'zh_CN']


def update_fq(tokens, fq):
    for w in tokens:
        fq[w] += 1
    return fq

def pretty(num): return "{:,}".format(num)


model = "lmqg/mbart-large-cc25-jaquad-qg"
language = 'ja'
dataset = 'vocabtrimmer/mc4_validation'
dataset_column: str = 'text'
dataset_name = "ja"
dataset_split = 'validation'
target_vocab_size = 10000
min_frequency = 2
chunk = 1000
cache_file_vocab = None
cache_file_frequency = None
overwrite = False
path_to_save = f'tmp.{os.path.basename(model)}'

dataset_name = language if dataset in ['mc4', 'vocabtrimmer/mc4_validation'] and dataset_name is None else dataset_name
logging.info(f"[DATASET INFO] dataset: {dataset}, name: {dataset_name}, split: {dataset_split}, column: {dataset_column}")
logging.info(f"[MINING INFO] language: {language}, model: {model}, chunk: {chunk}")

if cache_file_frequency is None:
    cache_file_frequency = f"{DEFAULT_CACHE_DIR}/vocab_mining/frequency.{dataset}.{dataset_column}.{dataset_name}.{dataset_split}.{model.replace('/', '_')}.json"
if cache_file_vocab is None:
    cache_file_vocab = f"{DEFAULT_CACHE_DIR}/vocab_mining/vocab.{dataset}.{dataset_column}.{dataset_name}.{dataset_split}.{model.replace('/', '_')}.{target_vocab_size}.{min_frequency}.json"


os.makedirs(os.path.dirname(cache_file_frequency), exist_ok=True)
os.makedirs(os.path.dirname(cache_file_vocab), exist_ok=True)
tokenizer = AutoTokenizer.from_pretrained(model)
config = AutoConfig.from_pretrained(model)
model = MBartForConditionalGeneration.from_pretrained(model, config=config)

os.makedirs(os.path.dirname(cache_file_frequency), exist_ok=True)
if not os.path.exists(cache_file_vocab):
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
with open(cache_file_vocab) as f:
    new_vocab = json.load(f)

vocab = dict(zip(tokenizer.all_special_tokens, tokenizer.all_special_ids))
vocab.update(new_vocab)
new_vocab_id = sorted(vocab.values())
new_vocab = list(vocab.keys())
logging.info(
    f'trimming vocabulary: {pretty(len(tokenizer.vocab))} (original) -> {pretty(len(new_vocab_id))} (target)')

################
# UPDATE MODEL #
################
logging.info("updating model")
# set input embedding
input_embedding = model.get_input_embeddings()
model.set_input_embeddings(torch.nn.Embedding.from_pretrained(input_embedding.weight[new_vocab_id]))

# set output embedding
output_embedding = model.get_output_embeddings()
if output_embedding is not None:
    new_weight = output_embedding.weight[new_vocab_id]
    new_bias = None
    if output_embedding.bias is not None:
        new_bias = output_embedding.bias[new_vocab_id]
    with torch.no_grad():
        linear = torch.nn.modules.linear.Linear(in_features=new_weight.shape[1], out_features=new_weight.shape[0],
                                                bias=output_embedding.bias is not None)
        linear.weight.copy_(new_weight)
        if new_bias is not None:
            linear.bias.copy_(new_bias)

    model.set_output_embeddings(linear)

# resize model vocab
model.config.vocab_size = len(new_vocab_id)
model.resize_token_embeddings(model.config.vocab_size)

# save to tem directory and load it
model.save_pretrained(path_to_save)
config = AutoConfig.from_pretrained(path_to_save)
model = MBartForConditionalGeneration.from_pretrained(path_to_save, config=config, ignore_mismatched_sizes=True)

####################
# UPDATE TOKENIZER #
####################
logging.info("updating tokenizer")

# update main vocab (except for the additionally added tokens, which NOT includes <pad>, <s>, </s>, <unk>)
model_state = json.loads(tokenizer.backend_tokenizer.model.__getstate__())
is_dict = False
if type(model_state['vocab']) is dict:
    is_dict = True
    model_state['vocab'] = list(model_state['vocab'].items())
new_state = []
for w, s in tqdm(model_state['vocab']):
    if w in new_vocab:
        new_state.append((w, s))
if is_dict:
    new_state = dict(new_state)
model_state['vocab'] = new_state
model_class = getattr(models, model_state.pop("type"))
tokenizer.backend_tokenizer.model = model_class(**model_state)

# update additional tokens (tokens added after pre-training won't be re-indexed above so need a dirty hack)
additional_special_tokens = [i for i in tokenizer.additional_special_tokens if i not in MBART_LANG_ID]
if len(additional_special_tokens) != 0:
    logging.info(f"updating additional_special_tokens of tokenizer")
    logging.info(f"num of add tokens: {len(additional_special_tokens)}")
    last_id = len(tokenizer.vocab) - 1 - len(additional_special_tokens)
    new_sp_token_index = {v: n + last_id + len(MBART_LANG_ID) + 1 for n, v in enumerate(additional_special_tokens)}
    _, _, _, path_added_token, _ = tokenizer.save_pretrained(path_to_save)
    with open(path_added_token, 'w') as f:
        json.dump(new_sp_token_index, f)
    tokenizer_2 = AutoTokenizer.from_pretrained(path_to_save)
