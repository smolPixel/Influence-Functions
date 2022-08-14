import argparse
import subprocess, os
from process_data import *
import random
import numpy as np
import torch
from Data.dataset import create_datasets
from Classifiers.classifiers import classifier
import yaml

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main(argdict):
    # run_lstm(argdict)
    train, dev, test=create_datasets(argdict)
    clas=classifier(argdict, train, dev, test)
    clas.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for VAE')
    # #General arguments on training

    parser.add_argument('--config_file', type=str, default='Configs/MNIST.yaml', help="Config file")
    args = parser.parse_args()
    args = args.__dict__

    print(args)
    stream=open(args['config_file'], "r")
    argsdict=yaml.safe_load(stream)
    main(argsdict)