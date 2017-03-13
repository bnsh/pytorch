#! /usr/bin/python

import torch
from torch.autograd.function import InplaceFunction

class rrelu(InplaceFunction):
	def __init__(self, l=0.125, u=1.0/3.0, train=True, inplace=False):
		super(rrelu, self).__init__()
		if (l < 0):
			raise ValueError("l must be >= 0: l={}, u={}".format(l,u))
		if (u < 0):
			raise ValueError("u must be >= 0: l={}, u={}".format(l,u))
		if (l > u):
			raise ValueError("l must be <= u: l={}, u={}".format(l, u))
		self.l = l
		self.u = u
		self.train = train
		self.inplace = inplace
		self.m = None

	def forward(self, inp):
		if self.inplace:
			self.mark_dirty(inp)
			out = inp
		else:
			out = inp.clone()

		if self.train:
			r = inp.new().resize_as_(inp).uniform_(self.l, self.u)
		else:
			r = inp.new().resize_as_(inp).fill_((self.l + self.u) / 2.0)
		zeros = torch.zeros(inp.size())
		lt0 = torch.mul(torch.lt(inp, zeros).type_as(inp), r)
		ge0 = torch.ge(inp, zeros).type_as(inp)
		self.m = (lt0 + ge0)
		out.mul_(self.m)
		return out

	def backward(self, grad_output):
		return grad_output.mul(self.m)


	def __repr__(self):
		return self.__class__.__name__ + '(l:%.7f, u:%.7f)' % (self.l, self.u)
