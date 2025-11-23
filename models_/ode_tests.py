import torch
import torch.nn as nn 
from torchdiffeq import odeint_adjoint as odeint 

### Phase 2
class ConcatFC(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(ConcatFC, self).__init__()
        self._layer = nn.Linear(dim_in, dim_out)
    def forward(self, t, x):
        return self._layer(x)


class ODEfunc_mlp(nn.Module): 

    def __init__(self, dim):
        super(ODEfunc_mlp, self).__init__()
        self.fc1 = ConcatFC(64, 64)
        self.act1 = torch.sin 
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = -1*self.fc1(t, x)
        out = self.act1(out)
        return out

### Phase 3
class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 5]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value    

print('SODEF')
odefunc = ODEfunc_mlp(0)
feature_layers = ODEBlock(odefunc)
print(feature_layers)



######################### NODE code
def norm(dim):
    return nn.BatchNorm2d(dim)

class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        # print('inside ConcatConv2d')
        # print('x', x.shape)
        # print('t', t)
        # print('x[:, :1, :, :]', x[:, :1, :, :].shape)
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1) # 
        # print('ttx', ttx.shape)
        """
        inside ConcatConv2d
        x torch.Size([1, 64, 6, 6])
        t tensor(0., device='cuda:0')
        x[:, :1, :, :] torch.Size([1, 1, 6, 6])
        ttx torch.Size([1, 65, 6, 6])
        """
        return self._layer(ttx)


class ODEfunc(nn.Module):

    def __init__(self, dim):
        super(ODEfunc, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 1]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

print('in Neural ODE code')
ode_layer = ODEBlock(ODEfunc(256))
print(ode_layer)


######################### My shee
def norm(dim):
    return nn.BatchNorm2d(dim)

class ODEfunc_time_invariant(nn.Module):

    def __init__(self, dim):
        super(ODEfunc_time_invariant, self).__init__()
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,)
        self.norm1 = norm(dim)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,)
        self.norm2 = norm(dim)
        self.relu2 = nn.ReLU()
        
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu(out)
        return out


class ODEBlock(nn.Module):

    def __init__(self, odefunc):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.integration_time = torch.tensor([0, 5]).float()

    def forward(self, x):
        self.integration_time = self.integration_time.type_as(x)
        out = odeint(self.odefunc, x, self.integration_time, rtol=1e-3, atol=1e-3)
        return out[1]

    @property
    def nfe(self):
        return self.odefunc.nfe

    @nfe.setter
    def nfe(self, value):
        self.odefunc.nfe = value

print('in Neural ODE code')
ode_layer = ODEBlock(ODEfunc(256))
print(ode_layer)