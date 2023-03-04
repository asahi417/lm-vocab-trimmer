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
  "ja": 125904,
  "ko":  73357,
  "ru": 147756,
  "fr": 131087,
  "it": 111056,
  "es": 131105,
  "de": 137617
}
```