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
import random
import ewc
import learner


torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed_all(42)

class Layer(torch.nn.Module):
    
    def __init__(self, input_size, output_size):
        
        super(Layer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights = Parameter(torch.Tensor(output_size, input_size))
        self.bias = Parameter(torch.Tensor(output_size))

        
        self.init_params()
        
    def init_params(self):
        
        stdv = 1. / math.sqrt(self.weights.size(1))
        self.weights.data.uniform_(-stdv, stdv)
        self.bias.data.uniform_(-stdv, stdv)  

        
    def forward(self, input):
        
        transform = F.linear(input, self.weights, bias = self.bias)  
        return transform 

class model_cl(torch.nn.Module):
    
    def __init__(self, Layer):

        super(model_cl, self).__init__()
        self.fc1 = Layer(28*28, 1000)   
        self.fc2 = Layer(1000, 10)
        
    def forward(self, x):
        x = x.view(-1, 28*28)
      
        x = self.fc1(x)
    
        x = F.relu(x, inplace = False)

        x = self.fc2(x)
        #return x
     
        return F.log_softmax(x)

def data_loader(DataLoaderObject, batch_size, train_class = None):
    
    data, target = list(DataLoaderObject)[0]
    idx = [i for i in range(len(data)) if target[i] in train_class]
    data_new, target_new = data[idx], target[idx]
    return torch.utils.data.DataLoader(list(zip(data_new, target_new)), batch_size = batch_size) 

if __name__ == "__main__":

	layer = Layer
	model = model_cl(layer)
	model.cuda()
	optimizer = optim.Adam(model.parameters(), lr=0.0005)
	permutations = list()
	task = 5

	for i in range(task):
      	
		permutations.append(torch.randperm(784).cuda())

	train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = 1, shuffle=True)

	test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size = 1, shuffle=True)

	ewc = ewc.EWC(model, train_loader, permutations, lamb = 0.5)
	CL = learner.CL(model, permutations, train_loader, test_loader, optimizer,  ewc = ewc)

	CL.train()
