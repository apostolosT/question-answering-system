import json
import os
import re
import string
from collections import Counter

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from src.models.document_reader_qa.model import QuAModel
from src.models.document_reader_qa.dataset import QuADataset
from src.models.document_reader_qa.model_lit import qaLightning


def train():

    torch.manual_seed(10)

    # For shuffle dataloader
    g = torch.Generator()
    g.manual_seed(10)

    train_path = "../../../datasets/squad/processed/train.pickle"
    train_set = QuADataset(train_path)
    val_path = "../../../datasets/squad/processed/val.pickle"
    val_set = QuADataset(val_path)
    # Reduce size for testing

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=16,
        shuffle=False,
    )

    validloader = torch.utils.data.DataLoader(
        val_set,
        batch_size=16,
    )
    with open("../../../datasets/squad/processed/idx2word.pickle", 'rb') as file:
        import pickle
        idx2word = pickle.load(file)
    evaluate_func = evaluate
    # Model
    HIDDEN_DIM = 128
    EMB_DIM = 300
    NUM_LAYERS = 3
    NUM_DIRECTIONS = 2
    DROPOUT = 0.3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = QuAModel(
        HIDDEN_DIM,
        EMB_DIM,
        NUM_LAYERS,
        NUM_DIRECTIONS,
        DROPOUT,
        device,
        "../../../datasets/squad/processed/weights-matrix.npy")

    litmodel = qaLightning(
        model,
        idx2word=idx2word,
        evaluate_func=evaluate_func,
        device=device,
        optimizer_lr=0.001)

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

    logger = TensorBoardLogger('tb_logs', name='dr_model')

    trainer = pl.Trainer(
        max_epochs=5,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        gradient_clip_val=10,
        devices=1,
        callbacks=[
            val_checkpoint,
            latest_checkpoint,
            early_stopping
        ],
        logger=logger,
        default_root_dir='../../../models'
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
    with open('../../../datasets/squad/raw/SQuAD-v1.1-dev.json', 'r', encoding='utf-8') as f:
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


def evaluate_single(predictions, answers, **kwargs):
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
      match exactly, 0 otherwise.
    : f1_score:
    '''
    assert len(predictions) == len(answers)
    f1 = exact_match = total = 0
    for key, value in predictions.items():
        prediction = value
        ground_truths = [answers[key]]

        exact_match += metric_max_over_ground_truths(
            exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score,
                                            prediction, ground_truths)

    total = len(predictions)
    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


def normalize_answer(s):
    '''
    Performs a series of cleaning steps on the ground truth and
    predicted answer.
    '''
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


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
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
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
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


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
