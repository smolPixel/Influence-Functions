from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import os
from sklearn.metrics import accuracy_score, recall_score, f1_score
import pytorch_lightning as pl
from utils import set_seed
import torch.nn.functional as F

class ConvNetClassifier(pl.LightningModule):

	def __init__(self, argdict, train, dev, test):
		super().__init__()
		self.argdict=argdict
		self.vocab_size=argdict['input_size']

		self.train_set = train
		self.dev_set = dev
		self.test_set = test
		self.init_model()
		self.loss_function = train.loss_function
		# self.conv1 = nn.Conv2d(3, 6, 5)
		# self.pool = nn.MaxPool2d(2, 2)
		# self.conv2 = nn.Conv2d(6, 16, 5)
		# self.fc1 = nn.Linear(16 * 5 * 5, 120)
		# self.fc2 = nn.Linear(120, 84)
		# self.fc3 = nn.Linear(84, 10)
		# self.loss_function=
		# print(self.model)

	def forward(self, input):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def init_model(self):
		set_seed(self.argdict['random_seed'])
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)


	def reset(self):
		self.init_model()

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
		return optimizer

	def get_logits(self, batch):
		input = batch['input']
		bs = input.shape[0]
		embed = self.embedding(input)
		embed = torch.mean(embed, dim=1)
		output = self.linear_layer(embed)
		return output

	def get_loss(self, batch):
		input=batch['input'].to(self.device)
		self.to(self.device)
		bs = input.shape[0]
		input_sequence = input.view(-1, self.argdict['input_size']).to(self.device).float()
		# embed=self.embedding(input)
		# embed=torch.mean(embed, dim=1)
		output=self.linear_layer(input_sequence)
		# best=torch.softmax(output, dim=-1)
		# pred=torch.argmax(best, dim=-1)
		# acc=accuracy_score(batch['label'].cpu(), pred.cpu())
		loss=self.loss_function(output, batch['label'])
		return loss

	def training_step(self, batch, batch_idx):
		input=batch['input']
		bs = input.shape[0]
		input_sequence = input.view(-1, self.argdict['input_size']).to('cuda').float()
		output=self.linear_layer(input_sequence)
		best=torch.softmax(output, dim=-1)
		pred=torch.argmax(best, dim=-1)
		acc=accuracy_score(batch['label'].cpu(), pred.cpu())
		loss=self.loss_function(output, batch['label'])
		self.log("Loss", loss, on_epoch=True, on_step=True, prog_bar=True, logger=False,
				 batch_size=bs)
		self.log("Acc Train", acc, on_epoch=True, on_step=False, prog_bar=True, logger=False,
				 batch_size=bs)
		return loss


	def validation_step(self, batch, batch_idx):
		input=batch['input']
		bs=input.shape[0]
		output=self.forward(input)
		best=torch.softmax(output, dim=-1)
		pred=torch.argmax(best, dim=-1)
		acc=accuracy_score(batch['label'].cpu(), pred.cpu())

		loss=self.loss_function(output, batch['label'])
		# self.log("Loss", loss, on_epoch=True, on_step=False, prog_bar=True, logger=False,
		#          batch_size=bs)
		fds
		self.log("Acc Dev", acc, on_epoch=True, on_step=False, prog_bar=True, logger=False,
				 batch_size=bs)
		return loss

	def validation_epoch_end(self, outputs):
		print("---\n")

	def train_model(self, train_set):
		self.trainer = pl.Trainer(gpus=1, max_epochs=self.argdict['nb_epoch_classifier'], precision=16, enable_checkpointing=False)
		# trainer=pl.Trainer(max_epochs=self.argdict['num_epochs'])
		train_loader = DataLoader(
			dataset=train_set,
			batch_size=64,
			shuffle=True,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)
		dev_loader = DataLoader(
			dataset=self.dev_set,
			batch_size=64,
			shuffle=False,
			# num_workers=cpu_count(),
			pin_memory=torch.cuda.is_available()
		)
		self.trainer.fit(self, train_loader, dev_loader)
		# fds

	#
	# def forward(self, inputs):
	# 	input_sequence = inputs.view(-1, self.argdict['input_size']).to(self.device)
	# 	input_sequence=input_sequence.float()
	# 	output = self.linear_layer(input_sequence)
	# 	return output
