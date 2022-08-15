"""Leave one out for finding the influence"""
from torch.utils.data import DataLoader
import torch
from utils import set_seed
from copy import deepcopy

class loo_influence():

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
		for j, batch in enumerate(data_loader):
			with torch.no_grad():
				loss = model.get_loss(batch).item()
			results_full[j] = loss

		influence=torch.zeros((len(train), len(dev)))


		#First thing first test the reset of the model and retraining
		# set_seed(self.argdict['random_seed'])
		model.reset()
		print(len(train))
		training_new=deepcopy(train)
		training_new.data.pop(0)
		training_new.reset_index()
		print(len(training_new))
		fds

