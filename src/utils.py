import pandas as pd
from tqdm import tqdm
tqdm.pandas()
import json
import numpy as np
import pickle
import os
import torch

import re
import string
def process_text(text):
    text = re.sub("(&#\d+;)", "", text)
    text = re.sub("([\/-])", " ", text)
    text = re.sub("(<.*?>)", "" ,text)
    text = re.sub("(^https?:\/\/\S+)", "", text)
    text = "".join([i for i in text if i not in string.punctuation + "…"])
    text = text.lower()
    return text

def process_corpus(corpus):
    _WORD_SPLIT = re.compile("([.,!?\"/':;)(])")
    def basic_tokenizer(sentence):
        words = []
        for space_separated_fragment in sentence.strip().split():
            words.extend(_WORD_SPLIT.split(space_separated_fragment))
        return [w.lower() for w in words if w != '' and w != ' ' and w not in string.punctuation]
    
    corpus = corpus.replace("\n", " ").split(" ")

def convert_lines(df, vocab, bpe, max_sequence_length):
    outputs = np.zeros((len(df), max_sequence_length))
    
    cls_id = 0
    eos_id = 2
    pad_id = 1

    for idx, row in tqdm(df.iterrows(), total=len(df)): 
        subwords = bpe.encode('<s> '+row.title2review+' </s>')
        input_ids = vocab.encode_line(subwords, append_eos=False, add_if_not_exist=False).long().tolist()
        if len(input_ids) > max_sequence_length: 
            input_ids = input_ids[:max_sequence_length] 
            input_ids[-1] = eos_id
        else:
            input_ids = input_ids + [pad_id, ]*(max_sequence_length - len(input_ids))
        outputs[idx,:] = np.array(input_ids)
    return outputs

def seed_everything(SEED):
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

