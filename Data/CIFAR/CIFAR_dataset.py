from torch.utils.data import Dataset
import pandas as pd
from nltk.tokenize import TweetTokenizer
from process_data import *
from torchtext.vocab import build_vocab_from_iterator
from torchvision import datasets, transforms
import copy
import matplotlib.pyplot as plt
#
class CIFAR10_dataset(Dataset):
	def __init__(self, data_ll):
		super().__init__()
		"""data: tsv of the data
		   tokenizer: tokenizer trained
		   vocabInput+Output: vocab trained on train"""
		self.data = {}
		self.loss_function=torch.nn.CrossEntropyLoss()
		self.categories=[0,1,2,3,4,5,6,7,8,9]
		index=0
		for i, row in enumerate(data_ll):
			input, label=row
			self.data[index] = {'input': input, 'label':label, 'target':input}
			index+=1

	def save_img(self, exo, name):
		# print(exo)
		# print(exo.shape)
		plt.imsave(name, exo.sequeeze(0).cpu().detach())
		# fds


	def reset_index(self):
		new_dat = {}
		for i, (j, dat) in enumerate(self.data.items()):
			new_dat[i] = dat
		self.data = new_dat

	def __len__(self):
		return len(self.data)

	def __getitem__(self, item):
		input = self.data[item]['input']
		label = self.data[item]['label']
		return {
			'input': np.asarray(input, dtype=float),
			'target': np.asarray(input, dtype=float),
			'label': label,
		}

	def shape_for_loss_function(self, logp, target):
		return logp, target.view(target.shape[0], -1).float().cuda()

	def iterexamples(self):
		for i, ex in self.data.items():
			yield i, ex

	def return_pandas(self):
		"""Return a pandas version of the dataset"""
		dict={}
		for i, ex in self.iterexamples():
			dict[i]={'sentence':ex['sentence'], 'label':ex['label']}
		return pd.DataFrame.from_dict(dict, orient='index')