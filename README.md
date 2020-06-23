# Faceted Domain Encoder
## Installation
1. Install poetry as dependency manager: `pip install --user poetry`
2. Navigate to project directory: `cd faceted-domain-encoder`
3. Install project dependencies: `poetry install`
4. Activate virtual environment: `poetry shell`
5. Download [spaCy](https://www.spacy.io) model: `python -m spacy download en`

## Run experiments
1. Activate virtual environment: `poetry shell`
2. Run experiment script from project directory, e.g.:
```
python experiments/sentence_similarity/medsts.py
```
3. Run a parameter sweep for an experiment, e.g.:
```bash
python experiments/sentence_similarity/medsts.py --multirun \
    encoder=gru,lstm,transformer \
    pooling=category_attention,max,mean \
    normalizer=corpus,document
```