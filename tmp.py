from transformers import AutoTokenizer

out = {}
for la, v_size in [("ja", 30000), ("ru", 30000), ("es", 105000), ("fr", 105000), ("it", 90000)]:
    tokenizer_a = AutoTokenizer.from_pretrained(f"vocabtrimmer/mt5-small-trimmed-{la}-{v_size}")
    tokenizer_b = AutoTokenizer.from_pretrained(f"vocabtrimmer/mt5-small-trimmed-{la}-{int(v_size-15000)}")
    out[la] = list(set(tokenizer_a.vocab.keys()) - set(tokenizer_b.vocab.keys()))


shared_tokens = None
for la, tokens in out.items():
    if shared_tokens is None:
        shared_tokens = set(tokens)
    else:
        shared_tokens = shared_tokens.intersection(set(tokens))
