import torch
import math
from torch import autograd
from torch import nn

# define data and target
input = autograd.Variable(torch.randn(3, 3), requires_grad = True)
print(input)
m = nn.Sigmoid()
print(m(input))

target = torch.FloatTensor([[0, 1, 1], [1, 1, 1], [0, 0, 0]])
print(target)

# according to formula
r11 = 0 * math.log(m(input)[0, 0].data.numpy()) + (1-0) * math.log((1 - m(input)[0, 0].data.numpy()))
r12 = 1 * math.log(m(input)[0, 1].data.numpy()) + (1-1) * math.log((1 - m(input)[0, 1].data.numpy()))
r13 = 1 * math.log(m(input)[0, 2].data.numpy()) + (1-1) * math.log((1 - m(input)[0, 2].data.numpy()))
r21 = 1 * math.log(m(input)[1, 0].data.numpy()) + (1-1) * math.log((1 - m(input)[1, 0].data.numpy()))
r22 = 1 * math.log(m(input)[1, 1].data.numpy()) + (1-1) * math.log((1 - m(input)[1, 1].data.numpy()))
r23 = 1 * math.log(m(input)[1, 2].data.numpy()) + (1-1) * math.log((1 - m(input)[1, 2].data.numpy()))
r31 = 0 * math.log(m(input)[2, 0].data.numpy()) + (1-0) * math.log((1 - m(input)[2, 0].data.numpy()))
r32 = 0 * math.log(m(input)[2, 1].data.numpy()) + (1-0) * math.log((1 - m(input)[2, 1].data.numpy()))
r33 = 0 * math.log(m(input)[2, 2].data.numpy()) + (1-0) * math.log((1 - m(input)[2, 2].data.numpy()))
r1 = -(r11 + r12 + r13) / 3
r2 = -(r21 + r22 + r23) / 3
r3 = -(r31 + r32 + r33) / 3
bceloss = (r1 + r2 + r3) / 3 
print(bceloss)

# PyTorch BCELoss
loss = nn.BCELoss()
print(loss(m(input), target))
