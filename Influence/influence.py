

class influence():

	def __init__(self, argdict):

		if argdict['influence_function'].lower()=="loo":
			from Influence.loo import loo_influence
			self.model=loo_influence(argdict)
		elif argdict['influence_function'].lower()=="blackbox":
			from Influence.blackBox import BlackBox_influence
			self.model=BlackBox_influence(argdict)
		elif argdict['influence_function'].lower()=="blackbox_group":
			from Influence.blackBox_group import BlackBox_influence_group
			self.model=BlackBox_influence_group(argdict)
		elif argdict['influence_function'].lower()=="blackbox_attack":
			from Influence.blackBoxAttack import BlackBox_Attack
			self.model=BlackBox_Attack(argdict)
		else:
			raise ValueError("Influence function not found")


	def calc_influence(self, model, train, dev):
		print(self.model.calc_influence(model, train, dev))

