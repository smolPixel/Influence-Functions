#! /usr/bin/env python3

import pytorch_influence_functions.pytorch_influence_functions as ptif
from pytorch_influence_functions.examples.train_influence_functions import load_model, load_data

if __name__ == "__main__":
    config = ptif.utils.get_default_config()
    model = load_model()
    trainloader, testloader = load_data()
    ptif.init_logging('logfile.log')
    ptif.calc_img_wise(config, model, trainloader, testloader)
