#! /usr/bin/python

import sys
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

rnnwidth = 2
rnnlayers = 1
batch_sz = 1024
sequence_sz = 8
lr=0.0002

class Parity(nn.Module):
	def __init__(self, middlesz, layers
		super(Parity, self).__init__()
		self.middlesz = middlesz
		self.layers = layers
		self.middle = nn.RNN(1,self.middlesz,self.layers)
		self.top = nn.Linear(self.middlesz, 1)

	def forward(self, inp):
		sequencesz = inp.data.size()[0]
		batchsz = inp.data.size()[1]
		mid, hidden = self.middle(inp)
		out = F.sigmoid(self.top(mid[sequencesz-1]))
		return out

def generate_data(batchsz, sequencesz):
	# Generate both inputs and targets.
	data = np.zeros((sequencesz, batchsz, 1))
	target = []
	for q in range(0, batchsz):
		piece = [[random.randint(0,1)] for _ in range(0, sequencesz)]
		data[:,q,:] = np.array(piece)

	target = np.sum(data,0)
	target = np.mod(target,2).reshape(target.shape[0],1)
	return Variable(torch.cuda.FloatTensor(data)), Variable(torch.cuda.FloatTensor(target))

def main():
	with torch.cuda.device(0):
		torch.manual_seed(12345)

		parity = Parity(rnnwidth, rnnlayers).cuda()
		optimizer = optim.Adam(parity.parameters(), lr=lr)
		criterion = nn.BCELoss()

		epoch = 0
		best_loss = None
		while best_loss is None or best_loss > 1e-5:
			optimizer.zero_grad()
			np_data, np_targets = generate_data(batch_sz, sequence_sz)
			output = parity.forward(np_data)
			accuracy = ((output>0.5) == (np_targets>0.5)).float().mean()
			loss = criterion(output, np_targets)
			if best_loss is None or best_loss > loss.data[0]:
				best_loss = loss.data[0]
				print("%d: New best: %.7f (%.7f)" % (epoch, best_loss, accuracy.data[0]))
			epoch += 1
			if epoch % 128 == 0:
				print(epoch, loss.data[0], accuracy.data[0])
			loss.backward()
			optimizer.step()

if __name__ == "__main__":
	main()
