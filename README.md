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
3. Evaluate all model architectures, e.g.:
    ```bash
    python experiments/sentence_similarity/medsts.py --multirun \
        encoder=gru,lstm,transformer \
        pooling=category_attention \
        normalizer=corpus,document
    ```
  
## Experiments
### Sentence Similarity Tasks
#### MedSTS
```
python experiments/sentence_similarity/medsts.py
```
#### BIOSSES
```
python experiments/sentence_similarity/medsts.py
```
### Text Classification
#### OHSUMED
```
python experiments/classification/ohsumed.py
```

#### Cancer Hallmarks
```
python experiments/classification/hallmarks.py
```

### Grep Outputs from Logs
```
 find outputs/ -type f | xargs grep -n 'MedSTS Test'
```