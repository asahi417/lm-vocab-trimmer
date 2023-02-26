import transformers
from vocabtrimmer.character_detector import filter_vocab

la = ['en', 'eu', 'ja', 'zh', 'ru', 'ko']
tokenizer = transformers.AutoTokenizer.from_pretrained("google/mt5-small")

stats = {}
for l in la:
    print(l)
    new_vocab = filter_vocab(tokenizer.vocab, l)
    stats[l] = len(new_vocab)
    print(new_vocab.keys())
    input()
print(stats)