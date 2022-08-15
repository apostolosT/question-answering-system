import pandas as pd
import pickle
from src.data import utils

WORD2IDX_PATH = 'datasets/squad/processed/word2idx.pickle'
TRAIN_SET_PATH = 'datasets/squad/raw/SQuAD-v1.1-train.json'
VAL_SET_PATH = 'datasets/squad/raw/SQuAD-v1.1-dev.json'
PROCESSED_TRAIN_SET_PATH = 'datasets/squad/processed/train.pickle'
PROCESSED_VAL_SET_PATH = 'datasets/squad/processed/val.pickle'

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

    # convert tokens in sentences to their respective idx
    print("Converting context tokens to their respective index, this may take a while...")
    train_df['context_ids'] = train_df.context.apply(utils.context_to_ids, word2idx=word2idx)
    val_df['context_ids'] = val_df.context.apply(utils.context_to_ids, word2idx=word2idx)

    train_df['question_ids'] = train_df.context.apply(utils.question_to_ids, word2idx=word2idx)
    val_df['question_ids'] = val_df.context.apply(utils.question_to_ids, word2idx=word2idx)

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

    train_df.to_pickle(PROCESSED_TRAIN_SET_PATH)
    val_df.to_pickle(PROCESSED_VAL_SET_PATH)

if __name__ == "__main__":
    make_dataset()
