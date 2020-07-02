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
### Semantic Similarity Tasks MedSTS & BIOSSES
```bash
chmod +x ./evaluate_semantic_similarity.sh
./evaluate_semantic_similarity.sh
```

### Word Importance per Category
Word importance per category on the Rolls-Royce Email and Case dataset.
#### Aviation
```bash
chmod +x ./evaluate_aviation_ablation.sh
./evaluate_aviation_ablation.sh
```

#### Medical
Word importance per category on the OHSUMED dataset. 
```bash
chmod +x ./evaluate_medical_ablation.sh
./evaluate_medical_ablation.sh
```
