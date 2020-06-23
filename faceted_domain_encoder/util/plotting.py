import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from IPython.core.display import display, Markdown

CATEGORY_PALLETTE = [
    '#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c', '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5', '#8c564b', '#c49c94',
    '#e377c2', '#f7b6d2', '#7f7f7f', '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5'
]


def short_category(text, max_tokens):
    tokens = text.split()
    suffix = tokens[-1]
    tokens = tokens[:-1]

    if len(tokens) > max_tokens:
        return ' '.join(tokens[:max_tokens]) + '...' + suffix

    return text


def plot_text(model, doc, filter_category=None, raw_html=False):
    indices, categories, length = model.processor(doc)
    markdown = ''

    for index, category in zip(indices, categories):
        if index == 0:
            continue

        category = category.item()
        token = model.processor.vocabulary.tokens[index]
        color = None
        weight = None

        if filter_category is not None:
            if filter_category == category:
                color = CATEGORY_PALLETTE[category]
                weight = round(model.processor.vocabulary.index2idf(index), 2)
        else:
            if category != -1:
                color = CATEGORY_PALLETTE[category]
                weight = round(model.processor.vocabulary.index2idf(index), 2)

        if color:
            markdown += f'<span style="color:{color}"><b>{token} [{weight}, C{category}]</b></span>'
        else:
            markdown += token

        markdown += ' '

    if raw_html:
        return markdown

    return display(Markdown(markdown))


def plot_attention(model, doc):
    model.eval()
    model.processor.vocabulary.freeze()
    index, category, length = model.processor(doc)

    # Move to gpu/cpu (handled by Lightning)
    batch_indices = index.unsqueeze(0)
    batch_categories = category.unsqueeze(0)
    batch_lengths = torch.tensor([length], dtype=torch.long)

    x, attention_weights = model.forward_batch(batch_indices, batch_categories, batch_lengths, attention=True)
    attention = attention_weights[0, :, :length].detach().cpu()
    all_tokens, _ = model.processor.vocabulary.index2sentence(index[:length])

    tokens = np.array(all_tokens)
    categories = np.array(model.hparams.graph.categories)
    categories = [short_category(c, 2) for c in categories]

    sns.set_context('paper')
    sns.set_style('whitegrid')
    plt.figure(figsize=(20, 20))
    ax = sns.heatmap(attention, xticklabels=tokens, yticklabels=categories, cmap='Blues')
    ax.set_title('Category Attention')
    return ax


def plot_category_weight(model, doc):
    model.eval()
    model.processor.vocabulary.freeze()

    index, category, length = model.processor(doc)

    # Move to device (handled by Lightning)
    batch_indices = index.unsqueeze(0)
    batch_categories = category.unsqueeze(0)
    batch_lengths = torch.tensor([length], dtype=torch.long)

    categories = list(model.hparams.graph.categories)
    categories = [short_category(c, 2) for c in categories]

    num_categories = model.hparams.graph.num_categories
    category_dims = model.hparams.model.document_embedding_dims // num_categories
    x, attention_weights = model.forward_batch(batch_indices, batch_categories, batch_lengths, attention=True)
    x = x.view(1, num_categories, category_dims)
    magnitudes = [x[0, c].norm().item() for c in range(num_categories)]

    sns.set_context('paper')
    sns.set_style('whitegrid')
    plt.figure(figsize=(4, 4))
    ax = sns.barplot(y=categories, x=magnitudes, palette=CATEGORY_PALLETTE)
    # ax.set_xticklabels(categories, rotation=90, horizontalalignment='right')
    ax.set_title('Category Embedding Magnitude')
    return ax

