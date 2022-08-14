import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
import pandas as pd
from nltk.tokenize import TweetTokenizer
from process_data import *
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
import copy

def create_datasets(argdict):
    if argdict['dataset'] in ['SST2', "SST100"]:
        from data.SST2.SST2Dataset import SST2_dataset
        #Textual dataset
        tokenizer=TweetTokenizer()

        train, dev, test=get_dataFrame(argdict)
        vocab = build_vocab_from_iterator((iter([tokenizer.tokenize(sentence) for sentence in list(train['sentence'])])),specials=["<unk>", "<pad>", "<bos>", "<eos>"])
        vocab.set_default_index(vocab["<unk>"])
        train=SST2_dataset(train, tokenizer, vocab, argdict)
        dev=SST2_dataset(dev, tokenizer, vocab, argdict)
        test=SST2_dataset(test, tokenizer, vocab, argdict)
        argdict['input_size']=train.vocab_size
        return train, dev, test
    elif argdict['dataset'] in ['MNIST']:
        #Image dataset
        from Data.MNIST.MNIST_dataset import MNIST_dataset
        train = datasets.MNIST(root='./mnist_data/', train=True, transform=transforms.ToTensor(), download=True)
        test = datasets.MNIST(root='./mnist_data/', train=False, transform=transforms.ToTensor(),
                                      download=False)
        train, dev=torch.utils.data.random_split(train, [55000, 5000])
        train=MNIST_dataset(train)
        dev=MNIST_dataset(dev)
        test=MNIST_dataset(test)
        argdict['input_size']=784
        return train, dev, test
    else:
        raise ValueError("dataset not found")
