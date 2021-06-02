import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

import numpy as np

### Define GPU
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

batch_size = 1000
n_iters = 5*10**6
epochs = int(n_iters/batch_size)
SyndTens = np.load('Syndromes2.npy')
ErrTens = np.load('Class2.npy')
SyndTest = np.load('Syndromes2Test.npy')
ErrTest = np.load('Class2Test.npy')

for i in range(5000000):
	if ErrTens[i][2] == 0. and ErrTens[i][3] == 0.:
		ErrTens[i] = torch.Tensor([1,0,0,0])
	elif ErrTens[i][2] == 1. and ErrTens[i][3] == 0.:
		ErrTens[i] = torch.Tensor([0,1,0,0])
	elif ErrTens[i][2] == 0. and ErrTens[i][3] == 1.:
		ErrTens[i] = torch.Tensor([0,0,1,0])
	elif ErrTens[i][2] == 1. and ErrTens[i][3] == 1.:
		ErrTens[i] = torch.Tensor([0,0,0,1])

for i in range(500000):
	if ErrTest[i][2] == 0. and ErrTest[i][3] == 0.:
		ErrTest[i] = torch.Tensor([1,0,0,0])
	elif ErrTens[i][2] == 1. and ErrTens[i][3] == 0.:
		ErrTest[i] = torch.Tensor([0,1,0,0])
	elif ErrTens[i][2] == 0. and ErrTens[i][3] == 1.:
		ErrTest[i] = torch.Tensor([0,0,1,0])
	elif ErrTens[i][2] == 1. and ErrTens[i][3] == 1.:
		ErrTest[i] = torch.Tensor([0,0,0,1])

SyndTensF = []
ErrTensF = []
for i in range(len(SyndTens)):
	SyndTensF.append(SyndTens[i][0:49])
	ErrTensF.append(ErrTens[i][0:49])
#np.save('syndtensf.npy',SyndTensF)
#np.save('errtensf.npy',ErrTensF)

SyndTens = torch.Tensor(SyndTensF).to(device)
ErrTens = torch.Tensor(ErrTensF).to(device)

SyndTestF = []
ErrTestF = []
for i in range(len(SyndTest)):
	SyndTestF.append(SyndTest[i][0:49])
	ErrTestF.append(ErrTest[i][0:49])
#np.save('syndtestf.npy',SyndTestF)
#np.save('errtestf.npy',ErrTestF)

SyndTest = torch.Tensor(SyndTestF).to(device)
ErrTest = torch.Tensor(ErrTestF).to(device)

### May have to move these to the GPU, though it may be implicit

dataset = TensorDataset(SyndTens,ErrTens)
testdata = TensorDataset(SyndTest,ErrTest)

train_loader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
test_loader = DataLoader(dataset=testdata,batch_size=batch_size, shuffle=True)

class FFN(nn.Module):
	def __init__(self, input_dim, hidden_dim, output_dim):
		super(FFN, self).__init__()
		# Linear function
		self.fc1 = nn.Linear(input_dim, 128) 

		# Non-linearity
		self.relu = nn.ReLU()

		self.sigmoid = nn.Sigmoid()

		self.tanh = nn.Tanh()

		#Layer 1
		self.fc1 = nn.Linear(49,128)

		#Layer 2
		self.fc2 = nn.Linear(128,128)

		#Layer 3
		self.fc3 = nn.Linear(128, 128)

		#Layer 4
		self.fc4 = nn.Linear(128,128)

		#Output
		self.fc5 = nn.Linear(128,4)



	def forward(self, x):
		# Linear function  # LINEAR
		out = self.fc1(x)
		out = self.relu(out)

		# Non-linearity  # NON-LINEAR
		out = self.fc2(out)
		out = self.relu(out)

		out = self.fc3(out)
		out = self.relu(out)

		# Linear function (readout)  # LINEAR
		out = self.fc4(out)
		out = self.relu(out)

		out = self.fc5(out)
		return out

### Model Dimensions
input_dim = 7*7
hidden_dim = 100
output_dim = 4

model = FFN(input_dim, hidden_dim, output_dim).to(device)

### Define Loss
criterion = nn.BCELoss()

### Define Optimizer
learning_rate=0.01
optimizer = torch.optim.Adam(model.parameters(), lr = learning_rate)

### Training
epochs = int(n_iters/batch_size)
iter1 = 0
#print(train_loader)
for epoch in range(epochs):
	if epoch % 3 == 0:
		learning_rate = 0.001
	trainingRunLoss = 0.
	for i, (syndromes, errors) in enumerate(train_loader):
		#print(syndromes.size())
		#syndromes = syndromes.view(-1,7*7).requires_grad_()
		#syndromes = syndromes.reshape(-1,7*7).requires_grad_()
		#errors = errors.reshape(-1,7*7)

		### Forward Pass
		outputs = model(syndromes.to(device))
		#sf = nn.Softmax(dim=1)
		outputs = torch.sigmoid(outputs)
		#outputs = torch.sigmoid(outputs)

		### Calculate Loss
		loss = criterion(outputs,errors.type(torch.float))
		if iter1 % 500 == 0:
			print(loss)
		trainingRunLoss += loss.item()

		### Compute gradients w/ backpropogation
		loss.backward()

		### Update parameters
		optimizer.step()

		### Clear gradient
		optimizer.zero_grad()

		iter1 += 1

		if iter1 % 1000 == 0:
			# Calculate Accuracy         
			RunningLoss = 0
			# Iterate through test dataset
			ctr = 0.
			Correct = 0.
			for syndromes, errors in test_loader:
				ctr += 1
				#print(errors.size())
				# Load images with gradient accumulation capabilities
				#print(test_loader.size())
				#syndromes = syndromes.reshape(-1, 7*7).requires_grad_()
				#print(test_loader.size())
				#errors = errors.reshape(-1,7*7)
				#print(syndromes.size())
				# Forward pass only to get logits/output
				outputs = model(syndromes)
				outputs = torch.sigmoid(outputs)
				#outputs = (outputs>0.5).float()
				loss = criterion(outputs,errors.type(torch.float))
				#outputs = sf(outputs)
				#sf = nn.Softmax(dim=1)
				#outputs = sf(outputs)
				binOut = (outputs>0.5).float()
				errors.float()
				#crct = torch.equal(binOut, errors)
				for ii in range(1000):
					if torch.equal(binOut[ii],errors[ii]):
						Correct += 1
				#print(crct)
				#Correct2 = (binOut == errors).float().sum()
				#Correct += Correct2
				#print(Correct2)
				#if crct != 4:
				#	Correct += 1
				#Correct += (binOut == errors).float().sum()
				#RunningLoss += loss.item()
			print(ctr)
			#print(loss)
			print('accuracy:')
			#print(Correct)
			print(Correct/(500*10**3))
	print(epoch)
	trainingRunLoss = trainingRunLoss/iter1






