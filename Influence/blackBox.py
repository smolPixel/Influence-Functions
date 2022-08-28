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
		test_loader = DataLoader(
			dataset=dev,
			batch_size=1,
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		train_loader= DataLoader(
			dataset=train,
			batch_size=32,
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)

		print("bru")
		for datapoint in test_loader:
			x=datapoint['input']
			y=datapoint['label']
			s_test=calc_s_test_single(model, x, y, train_loader)
			#Now that we have the s_test for the test point, we can calculate the influence of each trainng point on it
			train_dataset_size = len(train_loader.dataset)
			influences = []
			for i in range(train):
				print(train_loader.dataset[i])
				fds
				z = train_loader.collate_fn([z])
				t = train_loader.collate_fn([t])
				if time_logging:
					time_a = datetime.datetime.now()
				grad_z_vec = grad_z(z, t, model, gpu=gpu)
				if time_logging:
					time_b = datetime.datetime.now()
					time_delta = time_b - time_a
					logging.info(f"Time for grad_z iter:"
								 f" {time_delta.total_seconds() * 1000}")
				tmp_influence = -sum(
					[
						####################
						# TODO: potential bottle neck, takes 17% execution time
						# torch.sum(k * j).data.cpu().numpy()
						####################
						torch.sum(k * j).data
						for k, j in zip(grad_z_vec, s_test_vec)
					]) / train_dataset_size
				influences.append(tmp_influence.cpu())
				display_progress("Calc. influence function: ", i, train_dataset_size)

			harmful = np.argsort(influences)
			helpful = harmful[::-1]