import torch.autograd as ta
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.utils.data
from torchvision import datasets, transforms
import torch.optim as optim
import math
import matplotlib
from matplotlib import pyplot as plt
import torch.distributions as tdist
import random
import os 
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(42)
import copy
torch.set_printoptions(profile="full")


class EWC():

	def __init__(self, model, train_loader, permutations, lamb):
		super(EWC, self).__init__()
		self.model = model
		self.train_loader = train_loader
		self.permutations = permutations
		self.fisher_dict_list= list()
		self.param_history = list()
		self.tasks = len(permutations)
		self.lamb = lamb
		self.task_count = 1

	def calculate_fisher(self):

		fisher_dict = {}

		for n, p in self.model.named_parameters():
			fisher_dict[n] = p.detach().clone().zero_()

		self.model.eval()

		# Estimate Fisher Information Matrix for batchsize = 1
		# Is dataset for training or for testing?
		print('hello, fisher is calculating for task ', self.task_count )
		for batch_idx, (data, target) in enumerate(self.train_loader):

			data, target = Variable(data.cuda()), Variable(target.cuda())
			data = data.view(-1, 28*28)
			
			data = data[:, self.permutations[(self.task_count - 1)]]
			output = self.model(data)

			criterion = nn.CrossEntropyLoss()

			loss = criterion(output, target)


			self.model.zero_grad()
			loss.backward()


			for n, p in self.model.named_parameters():

				if p.requires_grad and p.grad is not None:

					fisher_dict[n] += p.grad.detach() ** 2

		print('hello, fisher is consolidating for task ', self.task_count )
		fisher_dict = {n: p/batch_idx for n,p in fisher_dict.items()}

		self.fisher_dict_list.append(fisher_dict)

		self.task_count += 1

		# consolidate parameters
		param_dict = {}
		for n, p in self.model.named_parameters():
			param_dict[n] = p.detach().clone()

		self.param_history.append(param_dict)

	def calculate_ewc_loss(self):

		losses = []

		for task in range(self.task_count-1):
			for n, p in self.model.named_parameters():
				
				if p.requires_grad:
					
					losses.append((self.fisher_dict_list[task][n]*(p-self.param_history[task][n])**2).sum())
		
		return (1./2)*sum(losses)




