"""Leave one out for finding the influence"""
from torch.utils.data import DataLoader
import torch

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

		print(results_full)