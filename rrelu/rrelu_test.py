#! /usr/bin/python

import sys
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from rrelu import rrelu

def test(mm, direction, inp_or_grad, label, expect, train):
	mm.train = train
	if direction == "forward":
		rv = mm.forward(inp_or_grad)
	elif direction == "backward":
		rv = mm.backward(inp_or_grad)
	if expect(rv):
		sys.stderr.write("pass: %s\n" % (label))
	else:
		print(rv)
		sys.stderr.write("fail: %s\n" % (label))

def close_to(v, epsilon):
	def func(rv):
		rv = rv.numpy()
		t = np.sqrt(np.power((rv-v), 2).sum())
		return t < epsilon
	return func

def uniform_checker(l, u):
	def cdf_uniform(x):
		x = np.sort(x)
		rv = np.zeros(x.shape)
		toolow = 0
		toohigh = x.shape[0]
		mxx = x.max()
		mnx = x.min()
		if mxx > u:
			toohigh = np.argmax(x > u)
			mxx = x[toohigh]
		if mnx < l:
			toolow = np.argmin(x < l)
			mnx = x[toolow]

		lo = (mnx - l) / (u-l)
		hi = (mxx - l) / (u-l)

		rv[toolow:toohigh] = np.linspace(lo, hi, (toohigh-toolow))
		return rv

	def func(rv):
		s, p = stats.kstest(rv.numpy(), cdf_uniform)
		return p > 0.999999
	return func

def main():
	torch.manual_seed(12345)
	sz = 1000
	epsilon = 1e-5
	mm = rrelu()
	positive_inp = torch.ones(sz)
	negative_inp = -torch.ones(sz)
	gradient = torch.ones(sz)

# defaults, training and positive
	test(mm, "forward",  positive_inp, " forward: rrelu(train=True)(torch.ones(%d)): expect all ones" % (sz), expect=close_to(1, epsilon), train=True)
	test(mm, "backward", gradient,     "backward: rrelu(train=True)(torch.ones(%d)): expect gradient=1" % (sz), expect=close_to(1, epsilon), train=True)

# defaults, training and negative
	test(mm, "forward",  negative_inp, " forward: rrelu(train=True)(torch.ones(%d)): expect uniform distribution between -1/3 and -1/8" % (sz), expect=uniform_checker(-1.0/3.0, -1./8.), train=True)
	test(mm, "backward", gradient,     "backward: rrelu(train=True)(torch.ones(%d)): expect uniform distribution between  1/8 and  1/3" % (sz), expect=uniform_checker( 1.0/8.0,  1./3.), train=True)

# defaults, eval and positive
	test(mm, "forward",  positive_inp, " forward: rrelu(train=False)(torch.ones(%d)): expect all ones" % (sz), expect=close_to(1, epsilon), train=False)
	test(mm, "backward", gradient,     "backward: rrelu(train=False)(torch.ones(%d)): expect gradient=1" % (sz), expect=close_to(1, epsilon), train=True)

# defaults, eval and negative
	test(mm, "forward",  negative_inp, " forward: rrelu(train=False)(torch.ones(%d)): expect all -11/48" % (sz), expect=close_to(-11./48., epsilon), train=False)
	test(mm, "backward", gradient,     "backward: rrelu(train=False)(torch.ones(%d)): expect gradient=11/48" % (sz), expect=close_to(11./48., epsilon), train=True)


	l = 0.1
	u = 0.9
	mm = rrelu(l=l, u=u)
# l=0.1, u=0.9, training and positive
	test(mm, "forward",  positive_inp, " forward: rrelu(l=%.7f, u=%.7f, train=True)(torch.ones(%d)): expect all ones" % (l, u, sz), expect=close_to(1, epsilon), train=True)
	test(mm, "backward", gradient,     "backward: rrelu(l=%.7f, u=%.7f, train=True)(torch.ones(%d)): expect gradient=1" % (l, u, sz), expect=close_to(1, epsilon), train=True)

# l=0.1, u=0.9, training and negative
	test(mm, "forward",  negative_inp, " forward: rrelu(l=%.7f, u=%.7f, train=True)(torch.ones(%d)): expect uniform distribution between %.7f and %.7f" % (l, u, sz, -u, -l), expect=uniform_checker(-u, -l), train=True)
	test(mm, "backward", gradient,     "backward: rrelu(l=%.7f, u=%.7f, train=True)(torch.ones(%d)): expect uniform distribution between %.7f and %.7f" % (l, u, sz, l, u), expect=uniform_checker(l, u), train=True)

# l=0.1, u=0.9, eval and positive
	test(mm, "forward",  positive_inp, " forward: rrelu(l=%.7f, u=%.7f, train=False)(torch.ones(%d)): expect all ones" % (l, u, sz), expect=close_to(1, epsilon), train=False)
	test(mm, "backward", gradient,     "backward: rrelu(l=%.7f, u=%.7f, train=False)(torch.ones(%d)): expect gradient=1" % (l, u, sz), expect=close_to(1, epsilon), train=False)

# l=0.1, u=0.9, eval and negative
	test(mm, "forward",  negative_inp, " forward: rrelu(l=%.7f, u=%.7f, train=False)(torch.ones(%d)): expect all %.7f" % (l, u, sz, (-u-l)/2.), expect=close_to((-u-l)/2., epsilon), train=False)
	test(mm, "backward", gradient,     "backward: rrelu(l=%.7f, u=%.7f, train=False)(torch.ones(%d)): expect uniform distribution between %.7f and %.7f" % (l, u, sz, l, u), expect=close_to((u+l)/2., epsilon), train=False)

	class Xor(nn.Module):
		def __init__(self, middlesz=2):
			super(Xor, self).__init__()
			self.middle = nn.Linear(2, middlesz)
			self.top = nn.Linear(middlesz, 1)

		def forward(self, inp):
			z = self.middle(inp)
			mid = rrelu()(z)
			out = F.sigmoid(self.top(mid))
			return out

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
