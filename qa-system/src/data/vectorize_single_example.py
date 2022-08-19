import pickle
from src.data import utils
from src import paths

with open(paths.WORD2IDX_PATH, 'rb') as file:
    word2idx = pickle.load(file)

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
