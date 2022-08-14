import argparse
import subprocess, os
from process_data import *
import random
import numpy as np
import torch
from Data.dataset import create_datasets
from Generator.Generator import generator
import yaml

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



def main(argdict):
    # run_lstm(argdict)
    train, dev, test=create_datasets(argdict)
    Gen = generator(argdict, train, dev, test)
    Gen.train()
    metrics=Gen.test()
    print(metrics)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Experiments for VAE')
    # #General arguments on training

    parser.add_argument('--config_file', type=str, default='Configs/SST-2/VAE.yaml', help="dataset you want to run the process on. Includes SST2, TREC6, FakeNews")
    args = parser.parse_args()
    args = args.__dict__

    print(args)
    stream=open(args['config_file'], "r")
    argsdict=yaml.safe_load(stream)
    main(argsdict)