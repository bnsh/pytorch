#! /usr/bin/python

import torch
from torch.autograd.function import Function

"""
	So, the input to the binary_stochastic should be a value "p" between [0,1]
	with probability p, it will output a 1 otherwise, it outputs a 0.
	Practically, this is accomplished simply by taking a random uniform
	the size of the input and seeing if the random uniform is less than
	the input. If true, we output a 1, if false we output a 0.

	The limits are _not_ enforced! If you pass a -2 to it, it will
	_always_ output 0 and if you pass a 2 to it, it will _always_ output
	a 1! Caveat emptor and all that.
"""

class binary_stochastic(Function):
	def __init__(self, train=True):
		super(binary_stochastic, self).__init__()
		self.train = train

	def forward(self, inp):
		if self.train:
			rnd = inp.clone()
			rnd.uniform_(0,1)
			out = rnd.lt(inp).type_as(inp)
		else:
			out = inp.ge(0.5).type_as(inp)
		return out

	def backward(self, grad_output):
		return grad_output
