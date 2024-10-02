# Zeroshot Tokenizer Transfer
## Note
- M(l): LM pre-trained in language `l`
- T(l): Downstream task in language `l`

## TODO
Target task is T(Y)
- Baseline 1: M(X) fine-tuned on T(X), testing with translation
- Baseline 2: M(X) fine-tuned on T(translate(X, Y)), testing with translation
- Baseline 3: M(X) fine-tuned on T(Y)


## Experiment
```shell

```