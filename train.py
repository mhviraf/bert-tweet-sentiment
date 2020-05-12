import config
import dataset
import os
import engine
import torch
import utils
import params
import pandas as pd
import torch.nn as nn
import numpy as np
from torch.optim import lr_scheduler
import argparse

from model import TweetModel
from sklearn import model_selection
from sklearn import metrics
import transformers
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

from apex import amp


def run(fold, model_name):
    dfx = pd.read_csv(config.TRAINING_FILE)

    df_train = dfx[dfx.kfold != fold].reset_index(drop=True)
    df_valid = dfx[dfx.kfold == fold].reset_index(drop=True)
    print(df_train.shape)
    print(df_valid.shape)
    train_dataset = dataset.TweetDataset(
        tweet=df_train.text.values,
        sentiment=df_train.sentiment.values,
        selected_text=df_train.selected_text.values
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config.TRAIN_BATCH_SIZE,
        num_workers=4
    )

    valid_dataset = dataset.TweetDataset(
        tweet=df_valid.text.values,
        sentiment=df_valid.sentiment.values,
        selected_text=df_valid.selected_text.values
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.VALID_BATCH_SIZE,
        num_workers=2
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device('cpu')
    print(f'training on {device}')
    model_config = transformers.RobertaConfig.from_pretrained(config.ROBERTA_PATH)
    model_config.output_hidden_states = True
    model = TweetModel(conf=model_config)
    model.to(device)

    num_train_steps = int(len(df_train) / config.TRAIN_BATCH_SIZE * config.EPOCHS)

    optimizer = AdamW(params.optimizer_params(model), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_train_steps)

    model, optimizer = amp.initialize(model, optimizer, opt_level="O1", verbosity=0)

    es = utils.EarlyStopping(patience=5, mode="max")
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler=scheduler)
        jaccard = engine.eval_fn(valid_data_loader, model, device)
        print(f"Jaccard Score = {jaccard}")
        print(f"Epoch={epoch}, Jaccard={jaccard}")
        es(jaccard, model, model_path=f"../gdrive/My Drive/tweet-sentiment-extraction/{model_name}-f{fold}.pt")
        if es.early_stop:
            print("Early stopping")
            break


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-fold', type=int, default=0)
    argparser.add_argument('-model_name', default='baseline')
    args = argparser.parse_args()

    run(fold=args.fold, model_name=args.model_name)
