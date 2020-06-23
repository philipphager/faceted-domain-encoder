import altair as alt
import numpy as np
import pandas as pd
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


def plot_scatter(frame, correlation):
    return alt.Chart(
        frame,
        title=f'MedSTS - Test Dataset {round(correlation, 4)}r',
        width=480,
        height=480
    ).mark_circle(opacity=0.5).encode(
        x=alt.X('score', title='MedSTS Annotator Score'),
        y=alt.Y('similarity', title='FDE Cosine Similarity'),
        tooltip=['pair_id', 'similarity', 'score']
    )
