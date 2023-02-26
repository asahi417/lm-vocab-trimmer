import json
import os
import logging
import shutil
from typing import List
from tqdm import tqdm
from math import prod

import torch
from tokenizers import models
from transformers import MT5ForConditionalGeneration, AutoConfig, pipeline, AutoTokenizer
from .character_detector import filter_vocab


def pretty(num): return "{:,}".format(num)


def get_cache_dir(root_dir):
    _id = 0
    while True:
        path = f"{root_dir}.{_id}"
        if not os.path.exists(path):
            break
        _id += 1
    return path


def show_parameter(target_model, log: bool = False):
    param_size_embedding = prod(target_model.encoder.embed_tokens.weight.shape) * 2
    param_size_full = sum(p.numel() for p in target_model.parameters())
    func = logging.info if log else print
    func(f"PARAMETER SUMMARY")
    func(f"\t*parameter size (full)  : {pretty(param_size_full)}")
    func(f"\t*parameter size (vocab) : {pretty(param_size_embedding)}")
    func(f"\t*parameter size (rest)  : {pretty(param_size_full - param_size_embedding)}")
    func(f"\t*ratio of vocab param   : {round(param_size_embedding / param_size_full * 100, 1)}%")
    return param_size_full, param_size_embedding


class MT5VocabTrimmer:

    def __init__(self, model_name: str):
        # load model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        self.model = MT5ForConditionalGeneration.from_pretrained(model_name, config=self.config)
        self.model_size_full, self.model_size_embedding = self.show_parameter(log=True)

    def text2text_generation(self, input_text: str):
        pipe = pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)
        return pipe(input_text)

    def show_parameter(self, log: bool = False): return show_parameter(self.model, log=log)

    def save_pretrained(self, path_to_save):
        self.model.save_pretrained(path_to_save)
        self.tokenizer.save_pretrained(path_to_save)

    def trim_vocab(self, language: str = None, vocab_to_keep: List = None, cache_dir: str = 'cache',
                   clean_cache: bool = True, path_to_save: str = None):
        """ Trim vocab of the model.

        :param language: language of tokens to keep in vocab
        :param vocab_to_keep: list of tokens to keep in vocab
        :param cache_dir: directory to save tokenizer and model
        :param clean_cache:
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
        self.model = MT5ForConditionalGeneration.from_pretrained(model_path, config=self.config)

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
            self.tokenizer.additional_special_tokens = []
            _, _, _, path_added_token, _ = self.tokenizer.save_pretrained(model_path)
            with open(path_added_token, 'w') as f:
                json.dump(new_sp_token_index, f)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model_size_full, self.model_size_embedding = self.show_parameter(log=True)
        if path_to_save is not None:
            self.save_pretrained(path_to_save)
            logging.info(f'model saved at `{path_to_save}`')

        if clean_cache:
            shutil.rmtree(model_path)
