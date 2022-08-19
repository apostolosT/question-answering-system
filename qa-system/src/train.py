import json
import os
import re
import string
from collections import Counter

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from configs.model import params, device

from src.data.utils import normalize_text

from src.models.qa_model.model import QuAModel
from src.models.qa_model.dataset import QuADataset
from src.models.qa_model.model_lit import QuALit

TRAIN_PATH = "../datasets/squad/processed/train.pickle"
VAL_PATH = "../datasets/squad/processed/val.pickle"
IDX2WORD_PATH = "../datasets/squad/processed/idx2word.pickle"
WEIGHTS_MATRIX_PATH = "../datasets/squad/processed/weights-matrix.npy"
RAW_VAL_SET_PATH = "../datasets/squad/raw/SQuAD-v1.1-dev.json"
LOGS_PATH = "../logs"


def train():

    torch.manual_seed(42)

    # For shuffle dataloader
    g = torch.Generator()
    g.manual_seed(42)

    train_set = QuADataset(TRAIN_PATH)
    val_set = QuADataset(VAL_PATH)

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=16,
        shuffle=False,
    )

    validloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=16,
    )
    with open(IDX2WORD_PATH, 'rb') as file:
        import pickle
        idx2word = pickle.load(file)

    evaluate_func = evaluate
    # Hyper Params
    lr = 0.001

    model = QuAModel(
        params
    )

    litmodel = QuALit(
        model,
        idx2word=idx2word,
        evaluate_func=evaluate_func,
        device=device,
        optimizer_lr=lr)

    val_checkpoint = ModelCheckpoint(
        filename="{epoch}-{step}-{val_loss:.1f}",
        monitor="val_loss",
        mode="min",
        save_top_k=-1
    )

    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="step",
        mode="max",
        every_n_train_steps=500,
        save_top_k=1
    )

    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=1
    )

    logger = TensorBoardLogger(LOGS_PATH, name='dr_model')

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=10,
        devices=1,
        callbacks=[
            val_checkpoint,
            latest_checkpoint,
            early_stopping
        ],
        logger=logger,
        default_root_dir=LOGS_PATH
    )

    trainer.fit(litmodel, trainloader, validloader)


def evaluate(predictions, **kwargs):
    '''
    Gets a dictionary of predictions with question_id as key
    and prediction as value. The validation dataset has multiple
    answers for a single question. Hence we compare our prediction
    with all the answers and choose the one that gives us
    the maximum metric (em or f1).
    This method first parses the JSON file, gets all the answers
    for a given id and then passes the list of answers and the
    predictions to calculate em, f1.
    :param dict predictions
    Returns
    : exact_match: 1 if the prediction and ground truth
      match exactly, 0 otherwise
    : f1_score:
    '''

    # TODO: Change to correct directory
    with open(RAW_VAL_SET_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    dataset = dataset['data']
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    continue

                ground_truths = list(map(lambda x: x['text'], qa['answers']))

                prediction = predictions[qa['id']]

                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)

                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total
    return exact_match, f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    '''
    Returns maximum value of metrics for predicition by model against
    multiple ground truths.
    :param func metric_fn: can be 'exact_match_score' or 'f1_score'
    :param str prediction: predicted answer span by the model
    :param list ground_truths: list of ground truths against which
                               metrics are calculated. Maximum values of
                               metrics are chosen.
    '''
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)

    return max(scores_for_ground_truths)


def f1_score(prediction, ground_truth):
    '''
    Returns f1 score of two strings.
    '''
    prediction_tokens = normalize_text(prediction).split()
    ground_truth_tokens = normalize_text(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    '''
    Returns exact_match_score of two strings.
    '''
    return (normalize_text(prediction) == normalize_text(ground_truth))


def epoch_time(start_time, end_time):
    '''
    Helper function to record epoch time.
    '''
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


if __name__ == "__main__":
    train()
