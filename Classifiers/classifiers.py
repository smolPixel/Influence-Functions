

class classifier():

	def __init__(self, argdict):
		self.argdict=argdict

		if self.argdict['classifier']=="Linear":
			from Classifiers.Linear import LinearClassifier
			self.model=LinearClassifier(self.argdict)
		else:
			raise ValueError("Classifier Not Found")