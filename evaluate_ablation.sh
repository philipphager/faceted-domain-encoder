#!/bin/sh
poetry shell


python experiments/ablation/ohsumed.py -m \
  pooling=graph_attention
