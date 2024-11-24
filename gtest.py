import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

a = torch.ones(1, 16, 2, 3, dtype=torch.float)
states = []
states.append(a)
states.append(a)
states.append(a)
states.append(a)
states.append(a)

action = np.array( [0, 0,    0,      0,     0, 0, 0, 1, 0], np.int8)

print(action[-2] == 1)

states = torch.stack(states, dim=0).detach()
states = states.squeeze(1)
print(np.shape(states))

sampled_indices = [1, 3]
sampled_states = states[sampled_indices]
sampled_states = sampled_states.squeeze(1)
print(sampled_states.shape)

class A():
    def __init__(self):
        self.all = torch.zeros(1, 2, 3, dtype=torch.float)
        
a = A()
print(a.all)