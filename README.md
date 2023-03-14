# [WIP] LM-Vocab-Trimmer: A Simple Model Compression by Trimming Embedding Matrix
The LM-Vocab-Trimmer a.k.a. `vocabtrimmer` is a model compression tool aiming at reducing the parameter size of multilingual LMs 
by trimming unused tokens from the embedding matrix.
This library assumes that you want to use or already fine-tuned a multilingual LM in a few specific languages, 
and other languages are not needed to be covered by the LM anymore.
Then, `vocabtrimmer` remove those tokens in the out-of-scope languages from the embedding matrix,

the input and the output embedding matrix .

Multilingual LMs ([mT5](https://arxiv.org/abs/2010.11934), [mBART](https://arxiv.org/abs/2001.08210), [XLM-R](https://arxiv.org/abs/1911.02116), etc) are

 is a pythob