from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from transformers import AutoTokenizer, AutoModelForSequenceClassification

dataset = "vocabtrimmer/mc4_validation"
dataset_config = "zh"
reference_model = "roberta-base"

reference_tokenizer = AutoTokenizer.from_pretrained(reference_model)
tokenizer = Tokenizer(models.Unigram())
tokenizer.normalizer = normalizers.NFKC()
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()
tokenizer.decoder = decoders.ByteLevel()
tokenizer.train_from_iterator()
# trainer = trainers.UnigramTrainer(
#     vocab_size=reference_tokenizer.vocab_size,
#     initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
#     special_tokens=tokenizer.all_special_tokens,
# )