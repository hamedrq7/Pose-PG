import torch 
import torch.nn as nn 
import numpy 
from advertorch.attacks import LinfSPSAAttack
import torch.nn.functional as F
bs = 64
images = torch.zeros((bs, 3, 32, 32))

class DummyModel(nn.Module): 
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.linear = nn.Linear(3*32*32, 10)
    
    def forward(self, x): 
        return self.linear(torch.flatten(x, 1))

dm = DummyModel()
loss = nn.CrossEntropyLoss()

spsa = LinfSPSAAttack(dm, 8/255., nb_sample=64, nb_iter=1, max_batch_size=64)
spsa.perturb(images, )
