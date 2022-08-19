import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# print(f"PROJECT_ROOT: {PROJECT_ROOT}")
BASE_DIR = os.path.dirname(PROJECT_ROOT)
# print(f"BASE_DIR: {BASE_DIR}")

RAW_VAL_SET_PATH = os.path.join(BASE_DIR, "datasets/squad/raw/SQuAD-v1.1-dev.json")
WORD2IDX_PATH = os.path.join(BASE_DIR, 'datasets/squad/processed/word2idx.pickle')
IDX2WORD_PATH = os.path.join(BASE_DIR, 'datasets/squad/processed/idx2word.pickle')
WORD_VOCAB_PATH = os.path.join(BASE_DIR, 'datasets/squad/processed/word_vocab.pickle')
MODEL_CKPT_PATH = os.path.join(BASE_DIR, "models/logs/dr_model/version_1/checkpoints/latest-epoch=4-step=27000.ckpt")
WEIGHTS_MATRIX_PATH = os.path.join(BASE_DIR, "datasets/squad/processed/weights-matrix.npy")
TRAIN_SET_PATH = os.path.join(BASE_DIR, "datasets/squad/processed/train.pickle")
VAL_SET_PATH = os.path.join(BASE_DIR, "datasets/squad/processed/val.pickle")
LOGS_PATH = os.path.join(BASE_DIR, "logs")
