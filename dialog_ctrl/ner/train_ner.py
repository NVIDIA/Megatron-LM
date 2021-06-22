
from src.config import get_params
from src.utils import init_experiment
from src.dataloader import get_dataloader
from src.model import EntityTagger
from src.trainer import NERTrainer

import torch
import numpy as np
from tqdm import tqdm
import random

def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def train_ner(params):
    # initialize experiment
    logger = init_experiment(params, logger_filename=params.logger_filename)

    # dataloader
    dataloader_train, dataloader_dev, dataloader_test = get_dataloader(params.model_name, params.batch_size, params.data_folder)

    # BERT-based NER Tagger
    model = EntityTagger(params)
    model.cuda()

    # trainer
    trainer = NERTrainer(params, model)
    trainer.train(dataloader_train, dataloader_dev, dataloader_test)


if __name__ == "__main__":
    params = get_params()

    random_seed(params.seed)
    train_ner(params)
