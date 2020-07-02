#!/bin/sh
poetry shell

# Case Dataset
python experiments/ablation/aviation_case.py -m \
    encoder=gru,lstm,transformer \
    pooling=category_attention \
    trainer.gpu=3

python experiments/ablation/aviation_case.py -m \
    encoder=lstm \
    pooling=self_attention \
    ablation.attention_map=False \
    trainer.gpu=3

# Email Dataset
python experiments/ablation/aviation_email.py -m \
    encoder=gru,lstm,transformer \
    pooling=category_attention \
    trainer.gpu=3

python experiments/ablation/aviation_email.py -m \
    encoder=lstm \
    pooling=self_attention \
    ablation.attention_map=False \
    trainer.gpu=3
