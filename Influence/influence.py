

class influence():

	def __init__(self, argdict):

		if argdict['influence_function'].lower()=="loo":
			from Influence.loo import loo_influence
			self.model=loo_influence(argdict)
		else:
			raise ValueError("Influence function not found")


	def calc_influence(self, model, train, dev):
		self.model.calc_influence(model, train, dev)

