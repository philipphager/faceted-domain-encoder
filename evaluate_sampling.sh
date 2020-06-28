#!/bin/sh
poetry shell

# Uniform vs Error Weighted
# What is the influence of embedding vs graph neighbors
python experiments/ablation/ohsumed.py -m \
  training.sampling=error_weighted,uniform \
  training.samples_per_document=8 \
  training.k_embedding=0,16,64 \
  training.k_graph=0,16,64 \
  ablation.attention_map=True \
  ablation.distance_map=False \

# Error Weighted vs Random
# What is the impact of the sample size on weighted vs random
python experiments/ablation/ohsumed.py -m \
  training.sampling=error_weighted,random \
  training.samples_per_document=2,8,32 \
  training.k_embedding=16 \
  training.k_graph=16 \
  ablation.attention_map=True \
  ablation.distance_map=False \
