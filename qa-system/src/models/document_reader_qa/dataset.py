from copyreg import pickle
import pandas as pd
import torch
from torch.utils.data import Dataset, dataset
import spacy

nlp = spacy.load('en_core_web_sm', disable=["tagger", "ner", "lemmatizer"])

class DocumentReaderQADataset(Dataset):
    def __init__(self, pickle_file_path):
        """
        Dataset for the DocumentReader QA system and its variations

        Args:
            pickle_file_path: The pickled train/val dataset
        """

        self.df = pd.read_pickle(pickle_file_path)
        self.max_context_length = max(len(ctx) for ctx in self.df.context_ids)
        self.max_question_length = max(len(ctx) for ctx in self.df.question_ids)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pass

    def get_span(self, text):
        text = nlp(text, disable=['parser', 'tagger', 'ner'])
        span = [(w.idx, w.idx + len(w.text)) for w in text]
        return span
