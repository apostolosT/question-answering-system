import pandas as pd
import numpy as np
import pickle
from src.data import utils

WORD2IDX_PATH = 'datasets/squad/processed/word2idx.pickle'
IDX2WORD_PATH = 'datasets/squad/processed/idx2word.pickle'
WORD_VOCAB_PATH = 'datasets/squad/processed/word_vocab.pickle'
TRAIN_SET_PATH = 'datasets/squad/raw/SQuAD-v1.1-train.json'
VAL_SET_PATH = 'datasets/squad/raw/SQuAD-v1.1-dev.json'
PROCESSED_TRAIN_SET_PATH = 'datasets/squad/processed/train.pickle'
PROCESSED_VAL_SET_PATH = 'datasets/squad/processed/val.pickle'
EMBEDDINGS_PATH = 'embeddings/glove/glove.6B.300d.txt'
WEIGHT_MATRIX_PATH = 'datasets/squad/processed/weights-matrix.npy'


def create_glove_matrix():
    '''
    Parses the glove word vectors text file and returns a dictionary with the words as
    keys and their respective pretrained word vectors as values.

    '''
    glove_dict = {}
    with open(EMBEDDINGS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            glove_dict[word] = vector

    return glove_dict

def create_word_embedding(glove_dict, word_vocab):
    '''
    Creates a weight matrix of the words that are common in the GloVe vocab and
    the dataset's vocab. Initializes OOV words with a zero vector.
    '''
    weights_matrix = np.zeros((len(word_vocab), 300))
    words_found = 0
    for i, word in enumerate(word_vocab):
        try:
            weights_matrix[i] = glove_dict[word]
            words_found += 1
        except Exception:
            pass
    return weights_matrix, words_found


def make_dataset():

    train_data = utils.load_json(TRAIN_SET_PATH)
    val_data = utils.load_json(VAL_SET_PATH)

    train_df = pd.DataFrame(utils.parse_data(train_data))
    val_df = pd.DataFrame(utils.parse_data(val_data))
    del train_data, val_data

    print("Normalizing whitespaces...")
    train_df.context = train_df.context.apply(utils.normalize_spaces)
    val_df.context = val_df.context.apply(utils.normalize_spaces)

    print("Building word vocabulary...")
    vocab_text = utils.gather_text_for_vocab([train_df, val_df])
    word2idx, idx2word, word_vocab = utils.build_word_vocab(vocab_text)

    glove_dict = create_glove_matrix()
    weights_matrix, words_found = create_word_embedding(glove_dict, word_vocab)
    print("Total words found in glove vocab: ", words_found)
    np.save(WEIGHT_MATRIX_PATH, weights_matrix)

    # convert tokens in sentences to their respective idx
    print("Converting context tokens to their respective index, this may take a while...")
    train_df['context_ids'] = train_df.context.apply(utils.context_to_ids, word2idx=word2idx)
    val_df['context_ids'] = val_df.context.apply(utils.context_to_ids, word2idx=word2idx)

    train_df['question_ids'] = train_df.question.apply(utils.question_to_ids, word2idx=word2idx)
    val_df['question_ids'] = val_df.question.apply(utils.question_to_ids, word2idx=word2idx)

    # get indices with tokenization errors and drop those indices
    train_err = utils.get_error_indices(train_df, idx2word)
    val_err = utils.get_error_indices(val_df, idx2word)

    train_df.drop(train_err, inplace=True)
    val_df.drop(val_err, inplace=True)

    # get start/end positions of answers from the context (the labels for training the qa models)
    print("Getting start/end index positions for the answers...")
    train_label_idx = train_df.apply(utils.index_answer, axis=1, idx2word=idx2word)
    val_label_idx = val_df.apply(utils.index_answer, axis=1, idx2word=idx2word)

    train_df['label_idx'] = train_label_idx
    val_df['label_idx'] = val_label_idx

    # save the processed dataset for later use
    print("Dumping data to '/datasets/squad/processed/'")
    with open(WORD2IDX_PATH, 'wb') as f:
        pickle.dump(word2idx, f)
    with open(IDX2WORD_PATH, 'wb') as f:
        pickle.dump(idx2word, f)
    with open(WORD_VOCAB_PATH, 'wb') as f:
        pickle.dump(word_vocab, f)

    train_df.to_pickle(PROCESSED_TRAIN_SET_PATH)
    val_df.to_pickle(PROCESSED_VAL_SET_PATH)

if __name__ == "__main__":
    make_dataset()
