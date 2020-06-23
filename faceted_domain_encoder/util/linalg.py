import torch
import torch.nn.functional as F


def split_categories(embeddings, category_dims):
    """
    Create a view of the embedding tensor with category dimension.
    From Documents x Embedding to Documents x Categories x Category Embedding
    """
    document_dims = embeddings.size(0)
    embedding_dims = embeddings.size(1)
    return embeddings.view(document_dims, category_dims, embedding_dims // category_dims)


def normalize(embeddings):
    """
    Normalize all vectors in embedding to magnitude one.
    """
    norm = embeddings.norm(p=2, dim=1, keepdim=True)
    embeddings = embeddings.div(norm)
    embeddings[embeddings.ne(embeddings)] = 0
    return embeddings


def pairwise_cosine_distance(x1, x2):
    x1 = normalize(x1)
    x2 = normalize(x2)
    return 1 - torch.mm(x1, x2.t())


def knns(embeddings, k_neighbours):
    """
    Return k-nearest-neighbours for each vector in embeddings.
    The resulting neighbours will not include the query vector itself.
    """
    embeddings = normalize(embeddings)
    dot_products = torch.matmul(embeddings, embeddings.t())
    cosine_similarity, neighbours = torch.topk(dot_products, k=k_neighbours + 1, dim=1)
    cosine_distance = 1 - cosine_similarity
    return neighbours[:, 1:], cosine_distance[:, 1:]


class CategoryDistance:
    def __init__(self, document_embedding_dims, num_categories):
        self.document_embedding_dims = document_embedding_dims
        self.num_categories = num_categories
        self.category_dims = self.document_embedding_dims // self.num_categories

    def __call__(self, x1, x2):
        if len(x1.shape) == 1:
            x1 = x1.unsqueeze(0)

        if len(x2.shape) == 1:
            x2 = x2.unsqueeze(0)

        x1 = self._split_categories(x1)
        x2 = self._split_categories(x2)
        return 1 - F.cosine_similarity(x1, x2, dim=-1)

    def _split_categories(self, x):
        # Move from (batch x embeddings) to (batch x  category x category embeddings)
        return x.view(-1, self.num_categories, self.category_dims)
