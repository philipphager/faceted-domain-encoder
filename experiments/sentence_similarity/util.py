import pandas as pd
import numpy as np

from scipy import stats
from torch import nn


def to_file(frame, file):
    tmp_df = pd.Series(np.column_stack((frame.sentence_1, frame.sentence_2)).flatten()).drop_duplicates()
    tmp_df.to_csv(file, index=False, header=False)


def sentence_similarity(model, frame):
    cosine = nn.CosineSimilarity(-1)
    sentence_1 = model.embed(frame.sentence_1.values).detach()
    sentence_2 = model.embed(frame.sentence_2.values).detach()
    return cosine(sentence_1, sentence_2)


def pearson_correlation(similarity, score):
    correlation, p = stats.pearsonr(similarity, score)
    return correlation
