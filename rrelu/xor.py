#! /usr/bin/python

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

class Xor(nn.Module):
	def __init__(self, middlesz=2):
		super(Xor, self).__init__()
		self.middle = nn.Linear(2, middlesz)
		self.top = nn.Linear(middlesz, 1)

	def forward(self, inp):
		z = self.middle(inp)
		mid = nn.RReLU()(z)
		out = F.sigmoid(self.top(mid))
		return out

def main():
	xor = Xor(middlesz=3)
	data = Variable(torch.Tensor([[0,0],[0,1],[1,0],[1,1]]))
	target = Variable(torch.Tensor([[0],[1],[1],[0]]))
	optimizer = optim.Adam(xor.parameters(), lr=0.001)
	criterion = nn.BCELoss()
	epoch = 0
	best_loss = None
	while best_loss is None or best_loss > 1e-5:
		optimizer.zero_grad()
		output = xor.forward(data)
		loss = criterion(output, target)
		if best_loss is None or best_loss > loss.data[0]:
			best_loss = loss.data[0]
		epoch += 1
		if epoch % 1024 == 0:
			print(epoch, loss.data[0])
			print(output)
		loss.backward()
		optimizer.step()
		

if __name__ == "__main__":
	main()
