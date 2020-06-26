import logging

import pandas as pd
import torch

logger = logging.getLogger(__name__)


def sample_documents(config, frame, num_samples):
    sample_dfs = []

    for i in range(len(config.graph.categories)):
        sample_df = frame[frame.document_category.map(lambda x: i in x)].sample(num_samples)
        sample_df['ablation_category'] = config.graph.categories[i]
        sample_dfs.append(sample_df)

    return pd.concat(sample_dfs)


def get_mean_average_precision(token_categories, category):
    num_tokens_in_category = (token_categories == category).sum()
    ideal_rank = torch.arange(num_tokens_in_category) + 1
    token_ranks = torch.arange(token_categories.size(0)) + 1

    actual_token_rank = token_ranks[token_categories == category]
    average_precision = (ideal_rank.float() / actual_token_rank.float())
    return average_precision.mean()


def get_document_ablations(index, category, length, unique_tokens=False, padding=0, invalid_category=-1):
    if unique_tokens:
        # Create a document of unique tokens only
        index, category = zip(*(set(zip(index.numpy(), category.numpy()))))
        index = torch.tensor(index)
        category = torch.tensor(category)
        length = index.size(0)

    ablation_index = index[:length].repeat(length, 1)
    ablation_category = category[:length].repeat(length, 1)
    ablation_length = torch.tensor([length] * length)

    ablation_index = ablation_index.fill_diagonal_(padding)
    ablation_category = ablation_category.fill_diagonal_(invalid_category)
    ablation_length = ablation_length - 1
    return (ablation_index, ablation_category, ablation_length), index, category, length


def get_tokens(model, index):
    return [model.processor.vocabulary.tokens[i] for i in index]


def get_category_names(config, category):
    return [config.graph.categories[c.item()] if c != -1 else None for c in category]


def ablation_study(model, config, frame, unique_tokens=False):
    documents = frame.text.values
    ablation_categories = frame.ablation_category.values
    category2index = {c: i for i, c in enumerate(config.graph.categories)}
    results = []

    for document_id, (document, ablation_category) in enumerate(zip(documents, ablation_categories)):
        ablation_category_id = category2index[ablation_category]
        document = documents[document_id]

        x = model.embed([document])
        model_input = model.processor(document)

        ablation_input, index, category, length = get_document_ablations(*model_input, unique_tokens)
        ablation, attention = model.forward_batch(*ablation_input, attention=True)

        # Select most relevant words by distance increase to original embedding when removed
        ablation_distances = model.category_distance(ablation, x)[:, ablation_category_id]
        sort_by_distance = torch.argsort(ablation_distances, descending=True)
        distance_index = index[sort_by_distance]
        distance_category = category[sort_by_distance]
        distance_mean_average_precision = get_mean_average_precision(distance_category, ablation_category_id)

        # Select most relevant words by attention weight
        attention = attention[0][ablation_category_id]
        sort_by_attention = torch.argsort(attention, descending=True)
        attention_index = index[sort_by_attention]
        attention_category = category[sort_by_attention]
        attention_mean_average_precision = get_mean_average_precision(attention_category, ablation_category_id)

        logger.info('Ablation: %s, %s, %s', document_id, distance_mean_average_precision, attention_mean_average_precision)

        results.append({
            'ablation_category_id': ablation_category_id,
            'ablation_category': ablation_category,
            'document_id': document_id,
            'distance_map': attention_mean_average_precision.item(),
            'attention_map': distance_mean_average_precision.item(),
            'distance_tokens': get_tokens(model, distance_index),
            'distance_categories': get_category_names(config, distance_category),
            'attention_tokens': get_tokens(model, attention_index),
            'attention_categories': get_category_names(config, attention_category),
        })

    return pd.DataFrame(results)
