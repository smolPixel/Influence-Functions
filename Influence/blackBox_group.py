"""Leave one out for finding the influence"""
from torch.utils.data import DataLoader
import torch
from utils import set_seed, calc_s_test_single, grad_z, display_progress
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt

class BlackBox_influence_group():
	"""Calculate influence of groups of training examples (in this case the classes) for a dev (or test) example following https://arxiv.org/pdf/1905.13289.pdf.
	The main difference with the solo version is that the change in self-loss is not additive and so has to be calculated separatly (self loss is define as the sum over the points of interest of
	the loss of each point). TODO: understand why it is not additive :/"""

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


		#Test
		influence_per_class=[0,0,0,0,0,
							 0,0,0,0,0]

		#We test one point at a time as in the paper
		for i, datapoint in enumerate(test_loader):
			influence_per_class = [0, 0, 0, 0, 0,
								   0, 0, 0, 0, 0]

			print(torch.cuda.memory_allocated())
			x=datapoint['input']
			train.save_img(x, 'test_exo.png')
			# plt.imsave('test_exo.png', x.cpu().detach().view(28, 28), cmap='gray_r')
			y=datapoint['label']
			#Find the s_test for the test point, invHessian * nabla(Loss(test_img, model params)), metionned in p.3. See function for more details
			#Code says that r*recursion depth = dataset size, however that is excrutiatingly slow.
			#########################
			# TODO: Experiment with various values of r depth and r to see when it stabilizes
			#########################
			s_test_vec=calc_s_test_single(model, x, y, train_loader, recursion_depth=self.argdict['recursion_depth'], r=20, gpu=0, scale=50000)
			# s_test_vec=calc_s_test_single(model, x, y, train_loader, recursion_depth=self.argdict['recursion_depth'], r=2, gpu=0, scale=50000)
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
			for i, batch in enumerate(train_loader_influence):
				input=batch['input'].cuda()
				label=batch['label'].cuda()
				grad_z_vec = grad_z(input, label, model, gpu=-1)
				tmp_influence = -sum(
					[
						####################
						# TODO: potential bottle neck, takes 17% execution time
						# torch.sum(k * j).data.cpu().numpy()
						####################
						#Eq 2
						torch.sum(k * j).data
						for k, j in zip(grad_z_vec, s_test_vec)
					]) / train_dataset_size
				influences.append(tmp_influence.item())
				influence_per_class[label.item()]+=tmp_influence.item()
				display_progress("Calc. influence function: ", i, train_dataset_size)
			# print(influences)
			# fds
			# helpful = np.argsort(influences)
			# harmful = helpful[::-1]
			# for i in range(5):
			# 	#helpful
			# 	ind=helpful[i]
			# 	img=train.data[ind]['input']
			# 	train.save_img(img, f'helpful_{i}.png')
			# 	# plt.imsave(f'helpful_{i}.png', img.cpu().detach().view(28, 28), cmap='gray_r')
			# 	#harmful
			# 	ind=harmful[i]
			# 	img=train.data[ind]['input']
			# 	train.save_img(img, f'harmful_{i}.png')
			# 	# plt.imsave(f'harmful_{i}.png', img.cpu().detach().view(28, 28), cmap='gray_r')
			print(y)
			print(influence_per_class)