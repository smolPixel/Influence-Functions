

class classifier():

	def __init__(self, argdict, train, dev, test):
		self.argdict=argdict

		if self.argdict['classifier'].lower()=="linear":
			from Classifiers.Linear import LinearClassifier
			self.model=LinearClassifier(self.argdict, train, dev, test)
		else:
			raise ValueError("Classifier Not Found")


	def train(self, train):
		return self.model.train_model(train)

	def get_loss(self, batch):
		return self.model.get_loss(batch)

	def get_logits(self, batch):
		return self.model.forward(batch)

	def reset(self):
		self.model.reset()