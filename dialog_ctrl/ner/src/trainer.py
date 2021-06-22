
import torch
import torch.nn as nn
from src.metrics import *
from src.dataloader import label_set, pad_token_label_id

import os
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger()

class NERTrainer(object):
    def __init__(self, params, model):
        self.params = params
        self.model = model

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params.lr)
        self.loss_fn = nn.CrossEntropyLoss()

        self.early_stop = params.early_stop
        self.no_improvement_num = 0
        self.best_dev_f1 = 0
    
    def train_step(self, X, y):
        self.model.train()

        preds = self.model(X)
        y = y.view(y.size(0)*y.size(1))
        preds = preds.view(preds.size(0)*preds.size(1), preds.size(2))

        self.optimizer.zero_grad()
        loss = self.loss_fn(preds, y)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, dataloader_train, dataloader_dev, dataloader_test):
        logger.info("Start NER training ...")
        for e in range(self.params.epoch):
            logger.info("============== epoch %d ==============" % e)
            loss_list = []
        
            pbar = tqdm(enumerate(dataloader_train), total=len(dataloader_train))
            for i, (X, y) in pbar:
                X, y = X.cuda(), y.cuda()

                loss = self.train_step(X, y)
                loss_list.append(loss)
                pbar.set_description("(Epoch {}) LOSS:{:.4f}".format(e, np.mean(loss_list)))

            logger.info("Finish training epoch %d. loss: %.4f" % (e, np.mean(loss_list)))

            logger.info("============== Evaluate epoch %d on Dev Set ==============" % e)
            f1_dev = self.evaluate(dataloader_dev)
            logger.info("Evaluate on Dev Set. F1: %.4f." % f1_dev)

            if f1_dev > self.best_dev_f1:
                logger.info("Found better model!!")
                self.best_dev_f1 = f1_dev
                self.no_improvement_num = 0
                self.save_model()
            else:
                self.no_improvement_num += 1
                logger.info("No better model found (%d/%d)" % (self.no_improvement_num, self.early_stop))

            if self.no_improvement_num >= self.early_stop:
                break
        
        logger.info("============== Evaluate on Test Set ==============")
        f1_test = self.evaluate(dataloader_test)
        logger.info("Evaluate on Test Set. F1: %.4f." % f1_test)
    
    def evaluate(self, dataloader):
        self.model.eval()

        pred_list = []
        y_list = []
        pbar = tqdm(enumerate(dataloader), total=len(dataloader))
        
        for i, (X, y) in pbar:
            y_list.extend(y.data.numpy()) # y is a list
            X = X.cuda()
            preds = self.model(X)
            pred_list.extend(preds.data.cpu().numpy())
        
        # concatenation
        pred_list = np.concatenate(pred_list, axis=0)   # (length, num_tag)
        pred_list = np.argmax(pred_list, axis=1)
        y_list = np.concatenate(y_list, axis=0)
        
        # calcuate f1 score
        pred_list = list(pred_list)
        y_list = list(y_list)
        lines = []
        for pred_index, gold_index in zip(pred_list, y_list):
            gold_index = int(gold_index)
            if gold_index != pad_token_label_id:
                pred_token = label_set[pred_index]
                gold_token = label_set[gold_index]
                lines.append("w" + " " + pred_token + " " + gold_token)
        results = conll2002_measure(lines)
        f1 = results["fb1"]

        return f1
    
    def save_model(self):
        """
        save the best model
        """
        saved_path = os.path.join(self.params.saved_folder, self.params.model_name+".pt")
        torch.save({
            "model": self.model,
        }, saved_path)
        logger.info("Best model has been saved to %s" % saved_path)
