import os

import matplotlib
import pandas as pd
import streamlit as st
import spacy
import torch
import torch.nn.functional as F
import urllib

from pathlib import Path

from faceted_domain_encoder import FacetedDomainEncoder
from faceted_domain_encoder.util.linalg import CategoryDistance
from faceted_domain_encoder.util.plotting import CATEGORY_PALLETTE


def init_spacy():
    spacy.cli.download('en')


def download_file(url, path):
    path = Path(path)

    if not path.exists():
        st.info(f'Downloading: {path}')
        urllib.request.urlretrieve(url, path)


@st.cache(allow_output_mutation=True)
def load_model(path):
    model_path = os.path.join(path, 'model.ckpt')
    return FacetedDomainEncoder.load_from_checkpoint(checkpoint_path=model_path)


@st.cache()
def load_documents(path):
    return pd.read_csv(path, sep='\n', header=None, names=['text'])


@st.cache(allow_output_mutation=True)
def load_embeddings(documents):
    return model.embed(documents)


def get_category_knns(embeddings, document_id, category_id, k=10):
    category_distance = CategoryDistance(512, 16)
    category_distances = category_distance(embeddings[document_id], embeddings)[:, category_id]
    sort_keys = torch.argsort(category_distances)
    return sort_keys[1:k + 1], category_distances[sort_keys][1:k + 1]


def get_knns(embeddings, document_id, k=10):
    x = embeddings[document_id]
    distances = 1 - F.cosine_similarity(x, embeddings, dim=-1)
    sort_keys = torch.argsort(distances)
    return sort_keys[1:k + 1], distances[sort_keys][1:k + 1]


def display_entity(model, doc, filter_category=None, verbose=False):
    indices, categories, length = model.processor(doc)
    markdown = ''

    for index, category in zip(indices, categories):
        if index == 0:
            continue

        category = category.item()
        token = model.processor.vocabulary.tokens[index]
        color = None

        if filter_category is not None:
            if filter_category == category:
                color = CATEGORY_PALLETTE[category]
        else:
            if category != -1:
                color = CATEGORY_PALLETTE[category]

        if verbose:
            markdown += f'<span style="color:{color}"><b>{token} ({index2category[category]})</b></span> ' if color else f'{token} '
        else:
            markdown += f'<span style="color:{color}"><b>{token}</b></span> ' if color else f'{token} '

    st.write(markdown, unsafe_allow_html=True)


def display_attention(model, document, filter_category=None):
    indices, categories, length = model.processor(document)

    batch_indices = indices.unsqueeze(0)
    batch_categories = categories.unsqueeze(0)
    batch_lengths = torch.tensor([length], dtype=torch.long)

    x, attention_weights = model.forward_batch(batch_indices, batch_categories, batch_lengths, attention=True)
    attention = attention_weights[0, :, :length].detach().cpu()

    markdown = ''

    for i, (index, category) in enumerate(zip(indices, categories)):
        token_category = filter_category

        if index == 0:
            continue

        if filter_category is None:
            token_category = attention[:, i].argmax()
            token_category = token_category.item()

        token = model.processor.vocabulary.tokens[index]
        weight = (attention[token_category][i] / attention[token_category].max()).item()
        color = CATEGORY_PALLETTE[token_category]
        rgb = matplotlib.colors.to_rgb(color)
        rgb = tuple([weight * c for c in rgb])
        color = matplotlib.colors.to_hex(rgb)
        font_weight = ((weight * 1000) // 100) * 100

        markdown += f'<span style="color:{color}; font-weight: {font_weight}">{token}</span> ' if color else f'{token} '

    st.write(markdown, unsafe_allow_html=True)


def display_document(model, document, category_id):
    if show_original == 'Preprocessed':
        if color_by == 'Entity':
            display_entity(model, document, filter_category=category_id)
        elif color_by == 'Attention':
            display_attention(model, document, filter_category=category_id)
        elif color_by == 'Entity (display category names)':
            display_entity(model, document, filter_category=category_id, verbose=True)
    else:
        st.write(document)


def modify_magnitude(embeddings, categories, document_id):
    embeddings = embeddings.view(-1, 16, 32)
    magnitudes = torch.tensor([embeddings[document_id, c].norm().item() for c in range(16)])
    st.sidebar.markdown(f'### Adjust Document Embedding')

    for category in categories:
        category_id = category2index[category]
        magnitude = magnitudes[category_id].item()
        new_magnitude = st.sidebar.slider(category, value=magnitude, min_value=0.001)

        embeddings[document_id, category_id] /= magnitude
        embeddings[document_id, category_id] *= new_magnitude

    return embeddings.view(-1, 512)


def format_option(x, n_words=5):
    excerpt = ' '.join(df.iloc[x].text.split()[:n_words])
    return f'{x}, {excerpt}...'


st.markdown('# Faceted Domain Encoder - Explorer 🔍')
st.markdown('*Julian Risch, Philipp Hager, Ralf Krestel, 2020.* Link to [paper](https://hpi.de/naumann/projects/web-science/deep-learning-for-text/multifaceted-embeddings.html)')

init_spacy()
download_file('https://www.dropbox.com/s/1ca4ckxlra0svqd/desc2020.xml?dl=1', 'data/graph/mesh/desc2020.xml')
download_file('https://www.dropbox.com/s/daiphzudbo5z1vh/graph.json?dl=1', 'data/graph/mesh/graph.json')
download_file('https://www.dropbox.com/s/sb7t2d1xxdr77hf/embedding.json?dl=1', 'data/graph/mesh/embedding.json')
download_file('https://www.dropbox.com/s/84qgm7ltf4mmb8v/model.ckpt?dl=1', 'data/ohsumed/model.ckpt')
download_file('https://www.dropbox.com/s/wnwr981hefu6pq3/test.txt?dl=1', 'data/ohsumed/test.txt')
download_file('https://www.dropbox.com/s/ps4keez6gecl2qu/vocabulary.json?dl=1', 'vocabulary.json')


model = load_model('./data/ohsumed')
model.eval()
categories = list(map(lambda x: x.split(' [')[0], model.hparams.graph.categories))
category2index = {c: i for i, c in enumerate(categories)}
index2category = {i: c for i, c in enumerate(categories)}

df = load_documents('./data/ohsumed/test.txt').head(100)
embeddings = load_embeddings(df.text.values)

document_id = st.selectbox('Select document', list(range(100)), format_func=format_option)
document = df.iloc[document_id].text
indices, document_categories, length = model.processor(document)

display_categories = [categories[c.item()] for c in document_categories.unique() if c.item() != -1]
category = st.selectbox('Select category', ['All'] + display_categories)
show_original = st.selectbox('Select document version', [
    'Preprocessed',
    'Original (not used by model, for intelligibility only)'
])

if show_original == 'Preprocessed':
    color_by = st.selectbox('Color by', ['Entity', 'Entity (display category names)', 'Attention'])

embeddings = modify_magnitude(embeddings, display_categories, document_id)

if category == 'All':
    neighbor_ids, distances = get_knns(embeddings, document_id)
    category_id = None
else:
    category_id = category2index[category]
    neighbor_ids, distances = get_category_knns(embeddings, document_id, category_id)

st.markdown(f'### Selected Document No. {document_id}')
display_document(model, document, category_id)

st.markdown('______')
st.markdown('### Top 10 Neighboring Documents')
for neighbor_id, distance in zip(neighbor_ids, distances):
    neighbor = df.iloc[neighbor_id.item()].text
    st.markdown(f'#### Document No. {neighbor_id.item()}')
    st.markdown(f'Cosine Distance: {distance:.3f}')
    display_document(model, neighbor, category_id)
