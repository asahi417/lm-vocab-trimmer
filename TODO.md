# TODO

## Experiment

Comparisons should include:
- FT: Fine-tuned
- FT-Trim: Fine-tuned -> Trimmed
- Trim-FT: Trimmed -> Fine-tuned
- Target vocab: 15k, 30k, 45k, 60k, 75k

Language: Non-English
Task & Model:
- mT5 (small): QG-Bench (7 languages)
- XLM-R (base): Multilingual Tweet Sentiment (ar, en, fr, de, it, pt, es) 
    
## Analysis
- Stats to track the changes
    * number of sentences in the dataset that were changed after trimming at tokenization
    * number of trimmed tokens appeared in the dataset
  
## Note
```python
mt5_max_vocab = {
  "ko":  73357,
  "it": 111056,
  "ja": 125904,
  "fr": 131087,
  "es": 131105,
  "de": 137617,
  "ru": 147756,
}
xlm_max_vocab = {
  "ar": 63460,
  "pt": 83153,
  "it": 84388,
  "fr": 106637,
  "es": 107364,
  "de": 113067
}
```
90000
105000
120000