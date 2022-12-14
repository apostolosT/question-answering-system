from collections import ChainMap

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class QuALit(pl.LightningModule):

    def __init__(self, model, optimizer_lr=0.001, device=None, idx2word=None, evaluate_func=None):
        super().__init__()
        self.model = model
        # print(self.model)
        self.lr = optimizer_lr
        self.dropout = nn.Dropout()
        self.device_ = device
        if not idx2word:
            raise NotImplementedError("idx2word not defined")
        if not evaluate_func:
            raise NotImplementedError(
                "Evaluation Function for validation not defined")
        self.idx2word = idx2word
        self.evaluate_func = evaluate_func
        # self.save_hyperparameters(ignore=['model'])

    def training_step(self, batch, batch_idx):
        context, question, context_mask, question_mask, label, ctx, ans, ids = batch

        preds = self.model(context, question, context_mask, question_mask)

        # forward pass, get the predictions
        start_pred, end_pred = preds
        # separate labels for start and end position
        start_label, end_label = label[:, 0], label[:, 1]
        # calculate loss
        loss = F.cross_entropy(start_pred, start_label) + \
            F.cross_entropy(end_pred, end_label)

        self.log('loss', loss.item(), on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        context, question, context_mask, question_mask, label, ctx, ans, ids = batch

        preds = self.model(context, question, context_mask, question_mask)

        p1, p2 = preds

        # for preds
        batch_size, c_len = p1.size()

        y1, y2 = label[:, 0], label[:, 1]

        loss = F.cross_entropy(p1, y1) + F.cross_entropy(p2, y2)

        self.log('val_loss', loss, on_step=True, on_epoch=True)

        predictions = {}
        answers = {}
        ls = nn.LogSoftmax(dim=1)
        mask = (torch.ones(c_len, c_len, device=self.device) * float('-inf')
                ).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
        score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
        score, s_idx = score.max(dim=1)
        score, e_idx = score.max(dim=1)
        s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()

        # unpack ans
        ans = ans[0]
        # stack predictions

        for i in range(batch_size):
            id = ids[i]
            pred = context[i][s_idx[i]:e_idx[i] + 1]
            pred = ' '.join([self.idx2word[idx.item()] for idx in pred])
            predictions[id] = pred
            answers[id] = ans[i]
        return (predictions, answers)

    def predict_step(self, batch, batch_idx):
        pl.seed_everything(1, workers=True)
        context, question, context_mask, question_mask = batch
        with torch.no_grad():
            preds = self.model(context, question, context_mask, question_mask)
            p1, p2 = preds
            # for preds
            batch_size, c_len = p1.size()

            ls = nn.LogSoftmax()
            mask = (torch.ones(c_len, c_len, device=self.device) * float('-inf')
                    ).tril(-1).unsqueeze(0).expand(batch_size, -1, -1)
            score = (ls(p1).unsqueeze(2) + ls(p2).unsqueeze(1)) + mask
            score, s_idx = score.max(dim=1)
            score, e_idx = score.max(dim=1)
            s_idx = torch.gather(s_idx, 1, e_idx.view(-1, 1)).squeeze()
            pred = context[0][s_idx:e_idx + 1]
            pred = ' '.join([self.idx2word[idx.item()] for idx in pred])
            return pred

    def training_epoch_end(self, training_step_outputs):
        loss = [x['loss'].item() for x in training_step_outputs]
        self.log('avg_epoch_loss', sum(loss) / len(loss))

    def validation_epoch_end(self, validation_step_outputs):
        # Unpack dicts
        predictions = dict(ChainMap(*[x[0] for x in validation_step_outputs]))
        answers = dict(ChainMap(*[x[1] for x in validation_step_outputs]))
        em, f1 = self.evaluate_func(predictions, answers=answers)

        print(f"\n em: {em}, f1: {f1} \n")
        self.log("val_em", em)
        self.log("val_f1", f1)

        return f1

    def configure_optimizers(self):
        return torch.optim.Adamax(self.model.parameters())
