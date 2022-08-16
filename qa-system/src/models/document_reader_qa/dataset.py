import os
import pandas as pd
import torch
from torch.utils.data import Dataset, dataset
import spacy

nlp = spacy.load('en_core_web_sm', disable=["tagger", "ner", "lemmatizer"])

class QuADataset(Dataset):
    def __init__(self, pickle_file_path):
        """
        Dataset for the QuAModel QA system and its variations

        Args:
            pickle_file_path: The pickled train/val dataset
        """

        self.df = pd.read_pickle(pickle_file_path)
        self.max_context_length = max(len(ctx) for ctx in self.df.context_ids)
        self.max_question_length = max(len(ctx) for ctx in self.df.question_ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        spans = []
        context_text = []
        answer_text = []
        batch = self.df.iloc[idx]

        context_text.append(batch.context)
        spans.append(self.get_span(batch.context))

        answer_text.append(batch.answer)

        # Fills the elements of the tensor with value 1 by selecting the indices in the order given in index.
        padded_context = torch.LongTensor(self.max_context_length).fill_(1)
        padded_context[:len(batch.context_ids)] = torch.LongTensor(batch.context_ids)

        padded_question = torch.LongTensor(self.max_question_length).fill_(1)
        padded_question[:len(batch.question_ids)] = torch.LongTensor(batch.question_ids)

        label = torch.LongTensor(batch.label_idx)
        context_mask = torch.eq(padded_context, 1)
        question_mask = torch.eq(padded_question, 1)

        ids = batch.id[0]  # index start

        return (
            padded_context,
            padded_question,
            context_mask,
            question_mask,
            label,
            context_text,
            answer_text,
            ids
        )

    def get_span(self, text):
        text = nlp(text, disable=['parser', 'tagger', 'ner'])
        span = [(w.idx, w.idx + len(w.text)) for w in text]
        return span

# Datagenerator functionality
if __name__ == '__main__':
    g = torch.Generator()
    g.manual_seed(10)
    cwd = os.getcwd()
    train_path = f"{cwd}/datasets/squad/processed/train.pickle"
    train_dataset = QuADataset(train_path)
    print(len(train_dataset))
    train_dataset.df = train_dataset.df[0:10]
    print(len(train_dataset))
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        generator=g,
    )
    for batch, data in enumerate(trainloader):
        print(batch)
        print(len(data[0]))
        a = data
        print(a[0].shape, a[1].shape, a[2].shape, a[3].shape, a[4].shape
              )
        break
