[![license](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://github.com/asahi417/lm-vocab-trimming/blob/master/LICENSE)
[![PyPI version](https://badge.fury.io/py/vocabtrimmer.svg)](https://badge.fury.io/py/vocabtrimmer)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/vocabtrimmer.svg)](https://pypi.python.org/pypi/vocabtrimmer/)
[![PyPI status](https://img.shields.io/pypi/status/vocabtrimmer.svg)](https://pypi.python.org/pypi/vocabtrimmer/)

# Vocabulary Trimming: An Efficient Multilingual Language Model Compression

<p align="center">
  <img src="https://raw.githubusercontent.com/asahi417/lm-vocab-trimming/master/assets/overview.png" width="400">
  <br><em> Figure 1: Three distinct QAG approaches. </em>
</p>


***Vocabulary Trimming (VT)*** is a model compression technique, which reduces a multilingual LM vocabulary to a 
target language by deleting irrelevant tokens from its vocabulary. 
LM to build monolingual LMs in any language covered by the multilingual LM. 
This library assumes that you want to use or already fine-tuned a multilingual LM in a few specific languages, 
and other languages are not needed to be covered by the LM anymore.
Then, `vocabtrimmer` remove those tokens in the out-of-scope languages from the embedding matrix,


### Motivation

pie.png
The bottleneck of a multilingual LM is its huge multilingual 
vocabulary, that results in a large model size and a high computational cost (eg. mT5 )
the input and the output embedding matrix .

Multilingual LMs ([mT5](https://arxiv.org/abs/2010.11934), [mBART](https://arxiv.org/abs/2001.08210), [XLM-R](https://arxiv.org/abs/1911.02116), etc) are

 is a pythob