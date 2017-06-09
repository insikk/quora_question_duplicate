# quora_question_duplicate

Various Tensorflow RNN models on Quora Question Duplicates Pair dataset.


# How to download data

* Download [Kaggle Quora Question Pairs](https://www.kaggle.com/c/quora-question-pairs/data).

For comparision
try train/dev/test parition here https://zhiguowang.github.io/#publications
with paper Zhiguo Wang, Wael Hamza, Radu Florian. Bilateral Multi-Perspective Matching for Natural Language Sentences. In eprint arXiv:1702.03814. 

# Requirements

* Quora Dataset
* Pretrained GloVe data
* Tensorflow 1.1 or higher
* Python3

# How to preprocess

It reads the raw data, then generate related dictionaries from pretrained GloVe word embeddings.


`python3 preprocess.py`


# How to Train

`python3 main.py --mode=train --noload`

# How to Test


load 6000's step saved weights to run test. 

`python3 main.py --load_step=6000`

# Result

Baseline. 840B 300d Glove, biLSTM encoder + 3 layer of neural net
> train=88.8, dev=80.35, test=80.6 @ 11k


Reproducing (Bowman 15) result
> 100d LSTM RNN
> train 84.8, dev 77.6
>
> 150d LSTM RNN
> 


Reproducing (Rocktaschel 16) result
> Conditional encoding, shared: Train, Dev, Test : 72

Reproducing LSTMN (Cheng 16) result
> ????


# TODO

Reproduce any result that are reported in the paper. My implementation seems lower than 3~10% of the reported performances. 

# Thanks to

* mLSTM (match-LSTM) implementation https://github.com/fuhuamosi/MatchLstm

* Keras SNLI baseline example https://github.com/Smerity/keras_snli

* Attention model for entailment on SNLI corpus implemented in Tensorflow https://github.com/shyamupa/snli-entailment

* Tensorflow implementation of Long-Short-Term-Memory Network for Natural Language Inference https://github.com/vsitzmann/snli-attention-tensorflow

* Code template from here. https://github.com/allenai/bi-att-flow


# Other resouces

Zhiguo Wang, Wael Hamza, Radu Florian. Bilateral Multi-Perspective Matching for Natural Language Sentences. In eprint arXiv:1702.03814. 
[code](https://github.com/zhiguowang/BiMPM)