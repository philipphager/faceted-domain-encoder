import torch
from torch.utils import data


def embed(model, frame, batch_size=100, num_workers=2):
    batches = []
    dataset = TextDataset(model.processor, frame.document.values)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    with torch.set_grad_enabled(False):
        for document, category, length in loader:
            batch = model.forward_batch(document, category, length)
            batches.append(batch)

    return torch.cat(batches)


class TextDataset(data.Dataset):
    def __init__(self, processor, documents):
        self.processor = processor
        self.documents = documents

    def __len__(self):
        return len(self.documents)

    def __getitem__(self, i):
        index, category, length = self.processor(self.documents[i])
        return index, category, length
