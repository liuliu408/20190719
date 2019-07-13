import torch
import os
import numpy as np
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
a = torch.rand((3,4))
print(a.size())
print(a)
b = torch.argmax(a, dim=0)
print(b)
print(b.size())
