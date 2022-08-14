

class classifier():

	def __init__(self, argdict, train, dev, test):
		self.argdict=argdict

		if self.argdict['classifier'].lower()=="linear":
			from Classifiers.Linear import LinearClassifier
			self.model=LinearClassifier(self.argdict, train, dev, test)
		else:
			raise ValueError("Classifier Not Found")


	def train(self):
		return self.model.train_model()