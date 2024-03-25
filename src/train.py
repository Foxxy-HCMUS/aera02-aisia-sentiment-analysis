import pandas as pd
from model import *
from tqdm import tqdm
tqdm.pandas()
from torch import nn
import json
import numpy as np
import pickle
import os
# from sklearn.preprocessing import scoreEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from transformers import *
import torch
import matplotlib.pyplot as plt
import torch.utils.data
import torch.nn.functional as F
import argparse
from transformers.modeling_utils import * 
from fairseq.data.encoders.fastbpe import fastBPE
from fairseq.data import Dictionary
# from vncorenlp import VnCoreNLP
import py_vncorenlp
from utils import *

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--train_path', type=str, default='/space/hotel/phit/personal/aera02-aisia/data/AERA02_AptitudeAssessment_Dataset_NLP_cleaned_vi.csv')
parser.add_argument('--dict_path', type=str, default="/space/hotel/phit/personal/aera02-aisia/src/phobert/PhoBERT_base_transformers/dict.txt")
parser.add_argument('--config_path', type=str, default="/space/hotel/phit/personal/aera02-aisia/src/phobert/PhoBERT_base_transformers/config.json")
# parser.add_argument('--rdrsegmenter_path', type=str, default="./vncorenlp/VnCoreNLP-1.2.jar", required=True)
parser.add_argument('--rdrsegmenter_path', type=str, default="/space/hotel/phit/personal/aera02-aisia/src/vncorenlp")
parser.add_argument('--pretrained_path', type=str, default='/space/hotel/phit/personal/aera02-aisia/src/phobert/PhoBERT_base_transformers/model.bin')
parser.add_argument('--max_sequence_length', type=int, default=256)
parser.add_argument('--batch_size', type=int, default=24)
parser.add_argument('--accumulation_steps', type=int, default=5)
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=69)
parser.add_argument('--lr', type=float, default=3e-5)
parser.add_argument('--ckpt_path', type=str, default='/space/hotel/phit/personal/aera02-aisia/model/PhoBERT')
parser.add_argument('--bpe-codes', default="/space/hotel/phit/personal/aera02-aisia/src/phobert/PhoBERT_base_transformers/bpe.codes",type=str, help='path to fastBPE BPE')

args = parser.parse_args()
bpe = fastBPE(args)

# py_vncorenlp.download_model(save_dir=args.rdrsegmenter_path)
# rdrsegmenter = VnCoreNLP(save_dir=args.rdrsegmenter_path, annotators="wseg", max_heap_size='-Xmx500m') 
rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=args.rdrsegmenter_path) 

seed_everything(69)

# Load model
config = RobertaConfig.from_pretrained(
    args.config_path,
    output_hidden_states=True,
    num_scores=1
)

model_bert = RobertaForSentimentAnalysis.from_pretrained(args.pretrained_path, config=config)
model_bert.cuda()

if torch.cuda.device_count():
    print(f"Training using {torch.cuda.device_count()} gpus")
    model_bert = nn.DataParallel(model_bert)
    tsfm = model_bert.module.roberta
else:
    tsfm = model_bert.roberta

# Load the dictionary  
vocab = Dictionary()
vocab.add_from_file(args.dict_path)

# Load training data
train_df = pd.read_csv(args.train_path).fillna("###")

# train_df = train_df.sample(100)
# train_df.reset_index(drop=True, inplace=True)

train_df.title2review = train_df.title2review.progress_apply(lambda x: ' '.join([' '.join(sent) for sent in rdrsegmenter.word_segment(x)]))
y = train_df.score.astype("int").values
X_train = convert_lines(train_df, vocab, bpe,args.max_sequence_length)

# Creating optimizer and lr schedulers
param_optimizer = list(model_bert.named_parameters())
no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
num_train_optimization_steps = int(args.epochs*len(train_df)/args.batch_size/args.accumulation_steps)
optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100, num_training_steps=num_train_optimization_steps)  # PyTorch scheduler
scheduler0 = get_constant_schedule(optimizer)  # PyTorch scheduler

loss_fn = nn.CrossEntropyLoss()

if not os.path.exists(args.ckpt_path):
    os.mkdir(args.ckpt_path)

splits = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=123).split(X_train, y))
for fold, (train_idx, val_idx) in enumerate(splits):
    print("Training for fold {}".format(fold))
    best_score = 0
    # if fold != args.fold:
    #     continue
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[train_idx],dtype=torch.long), torch.tensor(y[train_idx],dtype=torch.long))
    valid_dataset = torch.utils.data.TensorDataset(torch.tensor(X_train[val_idx],dtype=torch.long), torch.tensor(y[val_idx],dtype=torch.long))
    tq = tqdm(range(args.epochs + 1))
    for child in tsfm.children():
        for param in child.parameters():
            if not param.requires_grad:
                print("whoopsies")
            param.requires_grad = False
    frozen = True
    for epoch in tq:
        preds_all = []
        labels_all = []
        if epoch > 0 and frozen:
            for child in tsfm.children():
                for param in child.parameters():
                    param.requires_grad = True
            frozen = False
            del scheduler0
            torch.cuda.empty_cache()

        val_preds = None
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
        avg_loss = 0.
        avg_accuracy = 0.

        optimizer.zero_grad()
        pbar = tqdm(enumerate(train_loader),total=len(train_loader),leave=False)
        for i,(x_batch, y_batch) in pbar:
            model_bert.train()
            y_pred = model_bert(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
            loss =  loss_fn(y_pred.cuda(),y_batch.cuda())
            loss = loss.mean()
            loss.backward()
            if i % args.accumulation_steps == 0 or i == len(pbar) - 1:
                optimizer.step()
                optimizer.zero_grad()
                if not frozen:
                    scheduler.step()
                else:
                    scheduler0.step()
            lossf = loss.item()
            pbar.set_postfix(loss = lossf)
            avg_loss += loss.item() / len(train_loader)

        model_bert.eval()
        pbar = tqdm(enumerate(valid_loader),total=len(valid_loader),leave=False)
        for i,(x_batch, y_batch) in pbar:
            y_pred = model_bert(x_batch.cuda(), attention_mask=(x_batch>0).cuda())
            # y_pred = y_pred.squeeze().detach().cpu().numpy()
            # val_preds = np.atleast_1d(y_pred) if val_preds is None else np.concatenate([val_preds, np.atleast_1d(y_pred)])
            val_preds = y_pred.argmax(dim=-1)
            labels_all.extend(y_batch.cpu().numpy())
            preds_all.extend(val_preds.cpu().numpy())
        # best_th = 0
        # score = f1_score(y[val_idx], val_preds > 0.5)
        # print(f"\nAUC = {roc_auc_score(y[val_idx], val_preds):.4f}, F1 score @0.5 = {score:.4f}")
        
         # metrics
        accuracy = accuracy_score(labels_all, preds_all)
        precision = precision_score(labels_all, preds_all, average='macro')
        recall = recall_score(labels_all, preds_all, average='macro')
        f1 = f1_score(labels_all, preds_all, average='macro')
        print(f"\nAccuracy = {accuracy:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}, F1 score = {f1:.4f}")
        if f1 >= best_score:
            torch.save(model_bert.state_dict(),os.path.join(args.ckpt_path, f"model_{fold}.bin"))
            best_score = f1