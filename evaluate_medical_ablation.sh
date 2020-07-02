#!/bin/sh
poetry shell

# Case Dataset
python experiments/ablation/ohsumed.py -m \
    encoder=gru,lstm,transformer \
    pooling=category_attention \
    trainer.gpu=2

python experiments/ablation/ohsumed.py -m \
    encoder=lstm \
    pooling=self_attention \
    ablation.attention_map=False \
    trainer.gpu=2