#!/bin/sh
poetry shell

python experiments/ablation/aviation_case.py -m \
  model.encoder=lstm \
  model.pooling=category_attention \
  model.normalizer=pass \
  training.sampling=error_weighted \
  training.samples_per_document=4 \
  training.k_embedding=2,8,16,32,64 \
  training.k_graph=2,8,16,32,64 \
  validation.sampling=error_weighted \
  validation.samples_per_document=4 \
  validation.k_embedding=32 \
  validation.k_graph=32 \
  ablation.attention_map=True \
  ablation.distance_map=True \
  ablation.file='ablation_sampling.csv' \
  ablation.file_unique='ablation_unique_sampling.csv'

