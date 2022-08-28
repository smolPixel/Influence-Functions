"""Leave one out for finding the influence"""
from torch.utils.data import DataLoader
import torch
from utils import set_seed, calc_s_test_single
from copy import deepcopy

class BlackBox_influence():
	"""Calculate influence of training examples for a dev (or test) example following https://arxiv.org/pdf/1703.04730.pdf"""

	def __init__(self, argdict):
		self.argdict=argdict

	def calc_influence(self, model, train, dev):
		"""Takes in: a fully trained model,  the training set, and the dev set for which to calculate the influence"""
		results_full = torch.zeros((len(dev)))
		data_loader = DataLoader(
			dataset=dev,
			batch_size=1,
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)


		for datapoint in data_loader:
			x=datapoint['input']
			y=datapoint['label']
			print(calc_s_test_single(model, x, y, train))
