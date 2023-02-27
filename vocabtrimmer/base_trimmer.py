import json
import os
import logging
import shutil
from typing import List
from tqdm import tqdm
from math import prod

import pandas as pd
import torch
from tokenizers import models
from huggingface_hub import Repository
from transformers import MT5ForConditionalGeneration, MBartForConditionalGeneration, AutoConfig, pipeline, \
    AutoTokenizer, AutoModelForMaskedLM, AutoModelForTokenClassification, AutoModelForSequenceClassification
from .character_detector import filter_vocab

__all__ = ("MT5VocabTrimmer", "MBartVocabTrimmer", "XLMRobertaVocabTrimmer")
os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
MBART_LANG_ID = ['ar_AR', 'cs_CZ', 'de_DE', 'en_XX', 'es_XX', 'et_EE', 'fi_FI', 'fr_XX', 'gu_IN', 'hi_IN', 'it_IT',
                 'ja_XX', 'kk_KZ', 'ko_KR', 'lt_LT', 'lv_LV', 'my_MM', 'ne_NP', 'nl_XX', 'ro_RO', 'ru_RU', 'si_LK',
                 'tr_TR', 'vi_VN', 'zh_CN']


def safe_rmtree(path): shutil.rmtree(path) if os.path.exists(path) else None


def pretty(num): return "{:,}".format(num)


def get_cache_dir(root_dir):
    _id = 0
    while True:
        path = f"{root_dir}.{_id}"
        if not os.path.exists(path):
            break
        _id += 1
    return path


def show_parameter(target_model, log: bool = False, is_encoder_decoder: bool = True):
    param_size_embedding = prod(target_model.get_input_embeddings().weight.shape)
    if is_encoder_decoder:
        param_size_embedding = param_size_embedding * 2
    param_size_full = sum(p.numel() for p in target_model.parameters())
    vocab_size = len(target_model.get_input_embeddings().weight)

    func = logging.info if log else print
    func(f"PARAMETER SUMMARY")
    func(f"\t*parameter size (full) : {pretty(param_size_full)}")
    func(f"\t*parameter size (vocab): {pretty(param_size_embedding)}")
    func(f"\t*parameter size (rest) : {pretty(param_size_full - param_size_embedding)}")
    func(f"\t*ratio of vocab param  : {round(param_size_embedding / param_size_full * 100, 1)}%")
    func(f"\t*vocab size            : {pretty(vocab_size)}")
    return param_size_full, param_size_embedding, vocab_size


def push_to_hub(model, source_model, tokenizer, repo_id: str):
    # push to hub
    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)
    safe_rmtree(os.path.basename(repo_id))

    # update other files
    link = f"https://huggingface.co/{source_model}"
    software_link = "[`vocabtrimmer`](https://github.com/asahi417/lm-vocab-trimmer)"
    stats = [
        {
            "model": source_model,
            "parameter_size_full": pretty(model.config.trimming_stats["parameter_size_full/raw"]),
            "parameter_size_embedding": pretty(model.config.trimming_stats["parameter_size_embedding/raw"]),
            "vocab_size": pretty(model.config.trimming_stats["vocab_size/raw"]),
            "compression_rate_full": 100,
            "compression_rate_embedding": 100,
        },
        {
            "model": repo_id,
            "parameter_size_full": pretty(model.config.trimming_stats["parameter_size_full/trimmed"]),
            "parameter_size_embedding": pretty(model.config.trimming_stats["parameter_size_embedding/trimmed"]),
            "vocab_size": pretty(model.config.trimming_stats["vocab_size/trimmed"]),
            "compression_rate_full": round(model.config.trimming_stats["compression_rate_full"], 2),
            "compression_rate_embedding": round(model.config.trimming_stats["compression_rate_embedding"], 2),
        }
    ]

    df = pd.DataFrame(stats)
    df.index = df.pop("model")
    readme = f"# Vocabulary Trimmed [{source_model}]({link}): `{repo_id}` \n" \
             f"This model is a trimmed version of [{source_model}]({link}) by {software_link}, a tool for trimming " \
             f"vocabulary of language models to compress the model size.\n" \
             "Following table shows a summary of the trimming process.\n\n"
    readme += df.T.to_markdown()
    repo = Repository(os.path.basename(repo_id), repo_id)
    with open(f"{os.path.basename(repo_id)}/README.md", "w") as f:
        f.write(readme)
    repo.push_to_hub()
    safe_rmtree(os.path.basename(repo_id))


def save_pretrained(model, tokenizer, path_to_save):
    model.save_pretrained(path_to_save)
    tokenizer.save_pretrained(path_to_save)


class MT5VocabTrimmer:

    def __init__(self, model_name: str):
        # load model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        if self.config.model_type == 'mt5':
            self.__model_class = MT5ForConditionalGeneration
        elif self.config.model_type == 'mbart':
            self.__model_class = MBartForConditionalGeneration
        else:
            raise ValueError(f"model type {self.config.model_type} is not supported.")
        self.model = self.__model_class.from_pretrained(model_name, config=self.config)
        self.param_size_full_raw, self.param_size_embedding_raw, self.vocab_size_raw = self.show_parameter(log=True)
        self.param_size_full_trimmed, self.param_size_embedding_trimmed, self.vocab_size_trimmed = None, None, None

    def push_to_hub(self, repo_id: str): push_to_hub(self.model, self.model_name, self.tokenizer, repo_id)

    def save_pretrained(self, path_to_save): save_pretrained(self.model, self.tokenizer, path_to_save)

    def text2text_generation(self, input_text: str):
        return pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)(input_text)

    def show_parameter(self, log: bool = False): return show_parameter(self.model, log=log)

    def trim_vocab(self, language: str = None, vocab_to_keep: List = None, cache_dir: str = 'cache', clean_cache: bool = True):
        """ Trim vocab of the model.

        :param language: language of tokens to keep in vocab
        :param vocab_to_keep: list of tokens to keep in vocab
        :param cache_dir: directory to save tokenizer and model
        :param clean_cache:
        """
        assert language is not None or vocab_to_keep is not None, "language or vocab_to_keep must be specified."
        vocab = dict(zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids))
        if language is not None:
            vocab.update(filter_vocab(self.tokenizer.vocab, language))
        if vocab_to_keep is not None:
            vocab.update({k: self.tokenizer.vocab[k] for k in vocab_to_keep})
        if self.config.model_type == 'mbart':
            vocab.update({k: self.tokenizer.vocab[k] for k in MBART_LANG_ID})
        new_vocab_id = sorted(vocab.values())
        new_vocab = list(vocab.keys())
        model_path = get_cache_dir(cache_dir)

        logging.info(f'trimming vocabulary: {pretty(len(self.tokenizer.vocab))} (original) -> {pretty(len(new_vocab_id))} (target)')
        logging.info(f"cache directory: {model_path}")

        ################
        # UPDATE MODEL #
        ################
        logging.info("updating model")

        # set input embedding
        input_embedding = self.model.get_input_embeddings()
        self.model.set_input_embeddings(torch.nn.Embedding.from_pretrained(input_embedding.weight[new_vocab_id]))

        # set output embedding
        output_embedding = self.model.get_output_embeddings()
        new_weight = output_embedding.weight[new_vocab_id]
        with torch.no_grad():
            linear = torch.nn.modules.linear.Linear(in_features=new_weight.shape[1], out_features=new_weight.shape[0], bias=output_embedding.bias)
            linear.weight.copy_(new_weight)
        self.model.set_output_embeddings(linear)

        # resize model vocab
        self.model.config.vocab_size = len(new_vocab_id)
        self.model.resize_token_embeddings(self.model.config.vocab_size)

        # save to tem directory and load it
        self.model.save_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = self.__model_class.from_pretrained(model_path, config=self.config)

        ####################
        # UPDATE TOKENIZER #
        ####################
        logging.info("updating tokenizer")

        # update main vocab (except for the additionally added tokens, which NOT includes <pad>, <s>, </s>, <unk>)
        model_state = json.loads(self.tokenizer.backend_tokenizer.model.__getstate__())
        new_state = []
        for w, s in tqdm(model_state['vocab']):
            if w in new_vocab:
                new_state.append((w, s))
        model_state['vocab'] = new_state
        model_class = getattr(models, model_state.pop("type"))
        self.tokenizer.backend_tokenizer.model = model_class(**model_state)

        # update additional tokens (tokens added after pre-training won't be re-indexed above so need a dirty hack)
        if len(self.tokenizer.additional_special_tokens) != 0:
            logging.info(f"updating additional_special_tokens of tokenizer")
            logging.info(f"num of add tokens: {len(self.tokenizer.additional_special_tokens)}")
            last_id = len(self.tokenizer.vocab) - 1 - len(self.tokenizer.additional_special_tokens)
            new_sp_token_index = {v: n + last_id + 1 for n, v in enumerate(self.tokenizer.additional_special_tokens)}
            _, _, _, path_added_token, _ = self.tokenizer.save_pretrained(model_path)
            with open(path_added_token, 'w') as f:
                json.dump(new_sp_token_index, f)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # update config
        self.param_size_full_trimmed, self.param_size_embedding_trimmed, self.vocab_size_trimmed = self.show_parameter(log=True)
        stats = {
            "parameter_size_full/raw": self.param_size_full_raw,
            "parameter_size_embedding/raw": self.param_size_embedding_raw,
            "vocab_size/raw": self.vocab_size_raw,
            "parameter_size_full/trimmed": self.param_size_full_trimmed,
            "parameter_size_embedding/trimmed": self.param_size_embedding_trimmed,
            "vocab_size/trimmed": self.vocab_size_trimmed,
            "compression_rate_full": self.param_size_full_trimmed / self.param_size_full_raw * 100,
            "compression_rate_embedding": self.param_size_embedding_trimmed / self.param_size_embedding_raw * 100,
        }
        self.model.config.update({"trimming_stats": stats})

        if clean_cache:
            safe_rmtree(model_path)


MBartVocabTrimmer = MT5VocabTrimmer


class XLMRobertaVocabTrimmer:

    def __init__(self, model_name: str):
        # load model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        if len(self.config.architectures) > 1:
            logging.warn(f"model {model_name} has multiple architectures: {self.config.architectures}")
        if self.config.architectures[0].endswith("TokenClassification"):
            self.__model_class = AutoModelForTokenClassification
        elif self.config.architectures[0].endswith("SequenceClassification"):
            self.__model_class = AutoModelForSequenceClassification
        elif self.config.architectures[0].endswith("MaskedLM"):
            self.__model_class = AutoModelForMaskedLM
        else:
            raise ValueError(f"model type {self.config.architectures} is not supported.")
        self.model = self.__model_class.from_pretrained(model_name, config=self.config)
        self.param_size_full_raw, self.param_size_embedding_raw, self.vocab_size_raw = self.show_parameter(log=True)
        self.param_size_full_trimmed, self.param_size_embedding_trimmed, self.vocab_size_trimmed = None, None, None

    def push_to_hub(self, repo_id: str): push_to_hub(self.model, self.model_name, self.tokenizer, repo_id)

    def text_classification(self, input_text: str):
        return pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)(input_text)

    def token_classification(self, input_text: str):
        return pipeline("token-classification", model=self.model, tokenizer=self.tokenizer)(input_text)

    def fill_mask(self, input_text: str):
        return pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)(input_text)

    def show_parameter(self, log: bool = False): return show_parameter(self.model, log=log, is_encoder_decoder=False)

    def save_pretrained(self, path_to_save): save_pretrained(self.model, self.tokenizer, path_to_save)

    def trim_vocab(self, language: str = None, vocab_to_keep: List = None, cache_dir: str = 'cache',
                   clean_cache: bool = True):
        """ Trim vocab of the model.

        :param language: language of tokens to keep in vocab
        :param vocab_to_keep: list of tokens to keep in vocab
        :param cache_dir: directory to save tokenizer and model
        :param clean_cache:
        :param path_to_save: path to save the trimmed model and tokenizer
        :return:
        """
        assert language is not None or vocab_to_keep is not None, "language or vocab_to_keep must be specified."
        vocab = dict(zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids))
        if language is not None:
            vocab.update(filter_vocab(self.tokenizer.vocab, language))
        if vocab_to_keep is not None:
            vocab.update({k: self.tokenizer.vocab[k] for k in vocab_to_keep})
        new_vocab_id = sorted(vocab.values())
        new_vocab = list(vocab.keys())
        model_path = get_cache_dir(cache_dir)

        logging.info(
            f'trimming vocabulary: {pretty(len(self.tokenizer.vocab))} (original) -> {pretty(len(new_vocab_id))} (target)')
        logging.info(f"cache directory: {model_path}")

        ################
        # UPDATE MODEL #
        ################
        logging.info("updating model")

        # set input embedding
        input_embedding = self.model.get_input_embeddings()
        self.model.set_input_embeddings(torch.nn.Embedding.from_pretrained(input_embedding.weight[new_vocab_id]))

        # set output embedding
        output_embedding = self.model.get_output_embeddings()
        if output_embedding is not None:
            new_weight = output_embedding.weight[new_vocab_id]
            new_bias = None
            if output_embedding.bias is not None:
                new_bias = output_embedding.bias[new_vocab_id]
            with torch.no_grad():
                linear = torch.nn.modules.linear.Linear(in_features=new_weight.shape[1],
                                                        out_features=new_weight.shape[0],
                                                        bias=output_embedding.bias is not None)
                linear.weight.copy_(new_weight)
                if new_bias is not None:
                    linear.bias.copy_(new_bias)

            self.model.set_output_embeddings(linear)

        # resize model vocab
        self.model.config.vocab_size = len(new_vocab_id)
        self.model.resize_token_embeddings(self.model.config.vocab_size)

        # save to tem directory and load it
        self.model.save_pretrained(model_path)
        self.config = AutoConfig.from_pretrained(model_path)
        self.model = self.__model_class.from_pretrained(model_path, config=self.config, ignore_mismatched_sizes=True)

        ####################
        # UPDATE TOKENIZER #
        ####################
        logging.info("updating tokenizer")

        # update main vocab (except for the additionally added tokens, which NOT includes <pad>, <s>, </s>, <unk>)
        model_state = json.loads(self.tokenizer.backend_tokenizer.model.__getstate__())
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
        self.tokenizer.backend_tokenizer.model = model_class(**model_state)

        # update additional tokens (tokens added after pre-training won't be re-indexed above so need a dirty hack)
        if len(self.tokenizer.additional_special_tokens) != 0:
            logging.info(f"updating additional_special_tokens of tokenizer")
            logging.info(f"num of add tokens: {len(self.tokenizer.additional_special_tokens)}")
            last_id = len(self.tokenizer.vocab) - 1 - len(self.tokenizer.additional_special_tokens)
            new_sp_token_index = {v: n + last_id + 1 for n, v in enumerate(self.tokenizer.additional_special_tokens)}
            _, _, _, path_added_token, _ = self.tokenizer.save_pretrained(model_path)
            with open(path_added_token, 'w') as f:
                json.dump(new_sp_token_index, f)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.param_size_full_trimmed, self.param_size_embedding_trimmed, self.vocab_size_trimmed = self.show_parameter(
            log=True)
        stats = {
            "parameter_size_full/raw": self.param_size_full_raw,
            "parameter_size_embedding/raw": self.param_size_embedding_raw,
            "vocab_size/raw": self.vocab_size_raw,
            "parameter_size_full/trimmed": self.param_size_full_trimmed,
            "parameter_size_embedding/trimmed": self.param_size_embedding_trimmed,
            "vocab_size/trimmed": self.vocab_size_trimmed,
            "compression_rate_full": self.param_size_full_trimmed / self.param_size_full_raw * 100,
            "compression_rate_embedding": self.param_size_embedding_trimmed / self.param_size_embedding_raw * 100,
        }
        self.model.config.update({"trimming_stats": stats})

        if clean_cache:
            safe_rmtree(model_path)
