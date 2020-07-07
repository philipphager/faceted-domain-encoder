#!/bin/sh
poetry shell

# Error Weighted vs Random
# What is the impact of the sample size on weighted vs random
python experiments/ablation/aviation_email.py -m \
  model.encoder=lstm \
  model.pooling=category_attention \
  model.normalizer=pass \
  training.sampling=error_weighted,random \
  training.samples_per_document=4 \
  training.k_embedding=32 \
  training.k_graph=32 \
  validation.sampling=error_weighted \
  validation.samples_per_document=4 \
  validation.k_embedding=32 \
  validation.k_graph=32 \
  ablation.attention_map=True \
  ablation.distance_map=True \
  ablation.file='ablation_sampling.csv' \
  ablation.file_unique='ablation_unique_sampling.csv'

python experiments/ablation/aviation_case.py -m \
  model.encoder=lstm \
  model.pooling=category_attention \
  model.normalizer=pass \
  training.sampling=error_weighted,random \
  training.samples_per_document=4 \
  training.k_embedding=2 \
  training.k_graph=32 \
  validation.sampling=error_weighted \
  validation.samples_per_document=4 \
  validation.k_embedding=32 \
  validation.k_graph=32 \
  ablation.attention_map=True \
  ablation.distance_map=True \
  ablation.file='ablation_sampling.csv' \
  ablation.file_unique='ablation_unique_sampling.csv'

python experiments/ablation/ohsumed.py -m \
  model.encoder=lstm \
  model.pooling=category_attention \
  model.normalizer=pass \
  training.sampling=error_weighted,random \
  training.samples_per_document=4 \
  training.k_embedding=32 \
  training.k_graph=32 \
  validation.sampling=error_weighted \
  validation.samples_per_document=4 \
  validation.k_embedding=32 \
  validation.k_graph=32 \
  ablation.attention_map=True \
  ablation.distance_map=True \
  ablation.file='ablation_sampling.csv' \
  ablation.file_unique='ablation_unique_sampling.csv'
