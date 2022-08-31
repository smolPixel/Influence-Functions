"""Leave one out for finding the influence"""
from torch.utils.data import DataLoader
import torch
from utils import set_seed, calc_s_test_single, grad_z, display_progress
from copy import deepcopy
import numpy as np

class BlackBox_influence():
	"""Calculate influence of training examples for a dev (or test) example following https://arxiv.org/pdf/1703.04730.pdf"""

	def __init__(self, argdict):
		self.argdict=argdict

	def calc_influence(self, model, train, dev):
		"""Takes in: a fully trained model,  the training set, and the dev set for which to calculate the influence"""
		results_full = torch.zeros((len(dev)))
		test_loader = DataLoader(
			dataset=dev,
			batch_size=1,
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		train_loader= DataLoader(
			dataset=train,
			batch_size=1,
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)



		#We test one point at a time as in the paper
		for i, datapoint in enumerate(test_loader):
			x=datapoint['input']
			y=datapoint['label']
			#Find the s_test for the test point, invHessian * nabla(Loss(test_img, model params)), metionned in p.3. See function for more details
			#Code says that r*recursion depth = dataset size
			s_test_vec=calc_s_test_single(model, x, y, train_loader, recursion_depth=5000, r=10,gpu=1)
			#Now that we have the s_test for the test point, we can calculate the influence of each trainng point on it
			train_dataset_size = len(train_loader.dataset)
			influences = []
			train_loader_influence = DataLoader(
				dataset=train,
				batch_size=1,
				shuffle=False,
				# num_workers=cpu_count(),
				pin_memory=torch.cuda.is_available()
			)
			for batch in train_loader_influence:
				input=batch['input']
				label=batch['label']
				grad_z_vec = grad_z(input, label, model, gpu=-1)
				tmp_influence = -sum(
					[
						####################
						# TODO: potential bottle neck, takes 17% execution time
						# torch.sum(k * j).data.cpu().numpy()
						####################
						torch.sum(k * j).data
						for k, j in zip(grad_z_vec, s_test_vec)
					]) / train_dataset_size
				influences.append(tmp_influence.item().cpu())
				display_progress("Calc. influence function: ", i, train_dataset_size)
			print(influences)
			fds
			harmful = np.argsort(influences)
			helpful = harmful[::-1]
			print(helpful)