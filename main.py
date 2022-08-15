import argparse
import subprocess, os
from process_data import *
import random
import numpy as np
import torch
from Data.dataset import create_datasets
from Classifiers.classifiers import classifier
from Influence.influence import influence
import yaml

from utils import set_seed


def main(argdict):
    # run_lstm(argdict)
    # set_seed(argdict['random_seed'])
    train, dev, test=create_datasets(argdict)
    clas=classifier(argdict, train, dev, test)
    clas.train()
    infl=influence(argdict)
    infl.calc_influence(clas, train, dev)

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