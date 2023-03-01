import json
import os
import logging
from typing import List
from tqdm import tqdm
from math import prod

import pandas as pd
import torch
from tokenizers import models
from huggingface_hub import Repository
from transformers import MT5ForConditionalGeneration, MBartForConditionalGeneration, AutoConfig, pipeline, \
    AutoTokenizer, AutoModelForMaskedLM, AutoModelForTokenClassification, AutoModelForSequenceClassification
from .util import safe_rmtree, pretty
from .vocab_miner import vocab_miner


os.environ["OMP_NUM_THREADS"] = "1"  # to turn off warning message
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # to turn off warning message
MBART_LANG_ID = ['ar_AR', 'cs_CZ', 'de_DE', 'en_XX', 'es_XX', 'et_EE', 'fi_FI', 'fr_XX', 'gu_IN', 'hi_IN', 'it_IT',
                 'ja_XX', 'kk_KZ', 'ko_KR', 'lt_LT', 'lv_LV', 'my_MM', 'ne_NP', 'nl_XX', 'ro_RO', 'ru_RU', 'si_LK',
                 'tr_TR', 'vi_VN', 'zh_CN']


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
            "parameter_size_full": pretty(model.config.vocabtrimmer['stats']["parameter_size_full/raw"]),
            "parameter_size_embedding": pretty(model.config.vocabtrimmer['stats']["parameter_size_embedding/raw"]),
            "vocab_size": pretty(model.config.vocabtrimmer['stats']["vocab_size/raw"]),
            "compression_rate_full": 100,
            "compression_rate_embedding": 100,
        },
        {
            "model": repo_id,
            "parameter_size_full": pretty(model.config.vocabtrimmer['stats']["parameter_size_full/trimmed"]),
            "parameter_size_embedding": pretty(model.config.vocabtrimmer['stats']["parameter_size_embedding/trimmed"]),
            "vocab_size": pretty(model.config.vocabtrimmer['stats']["vocab_size/trimmed"]),
            "compression_rate_full": round(model.config.vocabtrimmer['stats']["compression_rate_full"], 2),
            "compression_rate_embedding": round(model.config.vocabtrimmer['stats']["compression_rate_embedding"], 2),
        }
    ]
    df = pd.DataFrame(stats)
    df.index = df.pop("model")
    readme = f"# Vocabulary Trimmed [{source_model}]({link}): `{repo_id}` \n" \
             f"This model is a trimmed version of [{source_model}]({link}) by {software_link}, a tool for trimming " \
             f"vocabulary of language models to compress the model size.\n" \
             "Following table shows a summary of the trimming process.\n\n"
    readme += df.T.to_markdown()
    readme += f"\n\n\nFollowing table shows the parameter used to trim vocabulary.\n\n " \
              f"{pd.DataFrame([model.config.vocabtrimmer['mining_config']]).to_markdown(index=False)}"
    repo = Repository(os.path.basename(repo_id), repo_id)
    with open(f"{os.path.basename(repo_id)}/README.md", "w") as f:
        f.write(readme)
    repo.push_to_hub()
    safe_rmtree(os.path.basename(repo_id))


def save_pretrained(model, tokenizer, path_to_save):
    model.save_pretrained(path_to_save)
    tokenizer.save_pretrained(path_to_save)


class VocabTrimmer:
    """ Vocabulary trimming for language localization of multilingual LM """

    def __init__(self, model_name: str):
        """ Vocabulary trimming for language localization of multilingual LM

        :param model_name: model name on huggingface or path to local model
        """
        # load model
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        if self.config.model_type in ['mt5', 'mbart']:
            logging.info("model is encoder-decoder LM")
            self.is_encoder_decoder = True
            if self.config.model_type == 'mbart':
                self.__model_class = MBartForConditionalGeneration
            else:
                self.__model_class = MT5ForConditionalGeneration
        else:
            logging.info("model is masked LM")
            self.is_encoder_decoder = False
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

    def push_to_hub(self, repo_id: str):
        push_to_hub(self.model, self.model_name, self.tokenizer, repo_id)

    def save_pretrained(self, path_to_save):
        save_pretrained(self.model, self.tokenizer, path_to_save)

    def text2text_generation(self, input_text: str):
        return pipeline("text2text-generation", model=self.model, tokenizer=self.tokenizer)(input_text)

    def text_classification(self, input_text: str):
        return pipeline("text-classification", model=self.model, tokenizer=self.tokenizer)(input_text)

    def token_classification(self, input_text: str):
        return pipeline("token-classification", model=self.model, tokenizer=self.tokenizer)(input_text)

    def fill_mask(self, input_text: str):
        return pipeline("fill-mask", model=self.model, tokenizer=self.tokenizer)(input_text)

    def show_parameter(self, log: bool = False):
        return show_parameter(self.model, is_encoder_decoder=self.is_encoder_decoder, log=log)

    def trim_vocab(self, language: str, path_to_save: str, dataset: str = 'mc4', dataset_column: str = 'text',
                   dataset_name: str = None, dataset_split: str = 'train', tokens_to_keep: List = None,
                   target_vocab_size: int = 70000, min_frequency: int = 2, chunk: int = 1000,
                   cache_file_vocab: str = None, cache_file_frequency: str = None):
        """ Vocabulary trimming along with vocabulary mining on corpus

        :param path_to_save: directly to save model
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
        :param tokens_to_keep: custom tokens to keep in vocabulary
        """

        # vocab mining
        dataset_name = language if dataset == 'mc4' and dataset_name is None else dataset_name
        new_vocab = vocab_miner(
            model=self.model_name,
            language=language,
            dataset=dataset,
            dataset_column=dataset_column,
            dataset_name=dataset_name,
            dataset_split=dataset_split,
            target_vocab_size=target_vocab_size,
            min_frequency=min_frequency,
            chunk=chunk,
            cache_file_frequency=cache_file_frequency,
            cache_file_vocab=cache_file_vocab
        )

        vocab = dict(zip(self.tokenizer.all_special_tokens, self.tokenizer.all_special_ids))
        vocab.update(new_vocab)
        if tokens_to_keep is not None:
            vocab.update({i: self.tokenizer.vocab[i] for i in tokens_to_keep})
        new_vocab_id = sorted(vocab.values())
        new_vocab = list(vocab.keys())
        logging.info(f'trimming vocabulary: {pretty(len(self.tokenizer.vocab))} (original) -> {pretty(len(new_vocab_id))} (target)')

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
                linear = torch.nn.modules.linear.Linear(in_features=new_weight.shape[1], out_features=new_weight.shape[0], bias=output_embedding.bias is not None)
                linear.weight.copy_(new_weight)
                if new_bias is not None:
                    linear.bias.copy_(new_bias)

            self.model.set_output_embeddings(linear)

        # resize model vocab
        self.model.config.vocab_size = len(new_vocab_id)
        self.model.resize_token_embeddings(self.model.config.vocab_size)

        # save to tem directory and load it
        self.model.save_pretrained(path_to_save)
        self.config = AutoConfig.from_pretrained(path_to_save)
        self.model = self.__model_class.from_pretrained(path_to_save, config=self.config, ignore_mismatched_sizes=True)

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
        additional_special_tokens = [i for i in self.tokenizer.additional_special_tokens if i not in MBART_LANG_ID]
        if len(additional_special_tokens) != 0:
            logging.info(f"updating additional_special_tokens of tokenizer")
            logging.info(f"num of add tokens: {len(additional_special_tokens)}")
            last_id = len(self.tokenizer.vocab) - 1 - len(additional_special_tokens)
            new_sp_token_index = {v: n + last_id + 1 for n, v in enumerate(additional_special_tokens)}
            _, _, _, path_added_token, _ = self.tokenizer.save_pretrained(path_to_save)
            with open(path_added_token, 'w') as f:
                json.dump(new_sp_token_index, f)
            self.tokenizer = AutoTokenizer.from_pretrained(path_to_save)

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
        self.model.config.update({"vocabtrimmer": {
            "stats": stats,
            "mining_config": {
                "language": language,
                "dataset": dataset,
                "dataset_column": dataset_column,
                "dataset_name": dataset_name,
                "dataset_split": dataset_split,
                "target_vocab_size": target_vocab_size,
                "min_frequency": min_frequency}}})

        # save model and tokenizer
        logging.info(f"saving model and tokenizer at {path_to_save}")
        self.save_pretrained(path_to_save)
