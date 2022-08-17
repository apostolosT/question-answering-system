import pickle
import re
from src.data import utils

WORD2IDX_PATH = 'datasets/squad/processed/word2idx.pickle'
IDX2WORD_PATH = 'datasets/squad/processed/idx2word.pickle'
WORD_VOCAB_PATH = 'datasets/squad/processed/word_vocab.pickle'


with open(WORD2IDX_PATH, 'rb') as f:
    word2idx = pickle.load(f)

with open(IDX2WORD_PATH, 'rb') as f:
    idx2word = pickle.load(f)

with open(WORD_VOCAB_PATH, 'rb') as f:
    word_vocab = pickle.load(f)

def vectorize_example(raw_context, raw_question):
    context = utils.normalize_spaces(raw_context)
    question = utils.normalize_spaces(raw_question)

    context_ids = utils.context_to_ids(context, word2idx)
    question_ids = utils.question_to_ids(question, word2idx)

    processed_example = {
        'context': context,
        'context_ids': context_ids,
        'question': question,
        'question_ids': question_ids,
    }
    return [processed_example]
