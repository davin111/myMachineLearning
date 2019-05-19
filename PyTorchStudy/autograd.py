#studying: https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html#sphx-glr-beginner-blitz-autograd-tutorial-py

import torch

print("[requires_grad, backward]")
x = torch.ones(2, 2, requires_grad=True)
print(x)
y = x + 2
print(y)
print(y.grad_fn)

z = y*y*3
out = z.mean()
print(z, out)

out.backward()
print(x.grad)
print(y.grad)
print(z.grad)

print()

print("[requires_grad_]")
a = torch.randn(2, 2)
a = ((a*3) / (a-1))
print(a.requires_grad)
a.requires_grad_()
print(a.requires_grad)
b = (a*a).sum()
print(b.grad_fn)

print()

print("[crazy things]")
x = torch.rand(3, requires_grad = True)
y = x*2
while y.data.norm() < 1000:
	y = y*2
print(y)

gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)
