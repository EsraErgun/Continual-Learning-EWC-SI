import torch.autograd as ta
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torchvision import datasets, transforms
import torch.distributions as tdist
import torch.nn as nn
import numpy as np
import torch.utils.data
import torch.optim as optim
import math
import random
import os 
import copy
import argparse

torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(42)

class CL():

	"""
	This class offers implementation of two continual learning methods for preventing catastrophic forgetting 
	in deep neural networks. 

	- Elastic Weight Consolidation (Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A., ... & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks. Proceedings of the national academy of sciences, 114(13), 3521-3526.)
	- Synaptic Intelligence (Zenke, F., Poole, B., & Ganguli, S. (2017, July). Continual learning through synaptic intelligence. In International conference on machine learning (pp. 3987-3995). PMLR.)

	"""

	def __init__(self, model, permutations, train_loader, test_loader, optimizer, ewc = None, si = None):
		
		super(CL, self).__init__()
		self.model = model
		self.permutations = permutations
		self.train_loader = train_loader
		self.test_loader = test_loader
		self.optimizer = optimizer
		self.ewc = ewc
		self.si = si
        
	def train(self):
		self.model.train()
		accuracy = 0

		for permutation in self.permutations:

			for epoch in range(3):

				if self.si:
					
					W = {}
					p_old = {}

					for n,p in self.model.named_parameters():
						W[n] = p.detach().clone().zero_()
						p_old[n] = p.data.clone()

				for batch_idx, (data, target) in enumerate(self.train_loader):
                    
					data, target = Variable(data.cuda()), Variable(target.cuda())
					data = data.view(-1, 28*28)
					data = data[:,permutation]

					self.optimizer.zero_grad()    
              
					output = self.model(data)
					_, indexes = torch.max(output, 1)
					accuracy += torch.mean((indexes == target).float())/5000.
					criterion = nn.CrossEntropyLoss()
					loss = criterion(output, target)

					if self.ewc and self.ewc.task_count > 1:

						loss_ewc = self.ewc.calculate_ewc_loss()
                        
						loss+=(5000)*loss_ewc

					loss.backward()
					self.optimizer.step()

					if batch_idx % 5000 == 0:

						print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, accuracy: {:.3f}'.format(epoch, batch_idx * len(data), len(self.train_loader.dataset),100. * batch_idx / len(self.train_loader), loss.data.item(), accuracy.data.item()))
						last_nonzero_acc = accuracy.data.item()
						accuracy = 0            

			if self.ewc:

				self.ewc.calculate_fisher()

			self.model.eval()

			test_loss = 0
			correct = 0

			for data, target in self.test_loader:
                
				data, target = data.cuda(), target.cuda()
				data, target = Variable(data, volatile=True), Variable(target)
				data = data.view(-1, 28*28)
				data = data[:, permutation]

				self.model.cuda()

				output = self.model(data)
				test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
				pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(target.data.view_as(pred)).cpu().sum()

			test_loss /= len(self.test_loader.dataset)

			print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
				test_loss, correct, len(self.test_loader.dataset),
				100. * correct / len(self.test_loader.dataset)))

		for permutation in self.permutations:

			self.model.eval()
			test_loss = 0
			correct = 0

			for data, target in self.test_loader:
            
				data, target = data.cuda(), target.cuda()
				data, target = Variable(data, volatile=True), Variable(target)
				data = data.view(-1, 28*28)

				data = data[:,permutation]
				self.model.cuda()
				output = self.model(data)

				test_loss += F.nll_loss(output, target, size_average=False).data.item() # sum up batch loss
				pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
				correct += pred.eq(target.data.view_as(pred)).cpu().sum()
			
			test_loss /= len(self.test_loader.dataset)

			print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
				test_loss, correct, len(self.test_loader.dataset),
				100. * correct / len(self.test_loader.dataset)))