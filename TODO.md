# TODO

- JS div on the test set always
- language-wise JS div
- rename Post/Pre-Trim to Post-FT/Pre-FT
- check the tokens that drops at the accuracy largerly


## Experiment

Comparisons should include:
- FT: Fine-tuned
- FT-Trim: Fine-tuned -> Trimmed
- Trim-FT: Trimmed -> Fine-tuned
- Target vocab: 15k, 30k, 45k, 60k, 75k
- Task: QA/QG/Sentiment

## Analysis
- Stats to track the changes
  * number of sentences in the dataset that were changed after trimming at tokenization
  * number of trimmed tokens appeared in the dataset
- Better to have?    
  * inference speed difference?
  * model size difference?



## MAYBE
- to remove the trimmed foundation model as it's unstable to fine-tune it.