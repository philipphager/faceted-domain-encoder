#!/bin/sh
poetry shell

# BIOSSES Baseline
#python experiments/sentence_similarity/biosses_baseline.py

# BIOSSES
python experiments/sentence_similarity/biosses.py -m \
  encoder=gru,lstm,transformer \
  normalizer=corpus,document,pass \
  pooling=category_attention,self_attention \


# MedSTS Baseline
python experiments/sentence_similarity/medsts_baseline.py

# MedSTS
python experiments/sentence_similarity/medsts.py -m \
  encoder=gru,lstm,transformer \
  normalizer=corpus,document,pass \
  pooling=category_attention,self_attention \
