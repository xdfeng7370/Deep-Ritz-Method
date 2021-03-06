import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim, autograd
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Try to solve the poisson equation:
'''  Solve the following PDE
-\Delta u(x) = 0, x\in (0,1)^10
u(x)=\sum_{k=1}^5x_{2k-1}x_{2k} on \partial \Omega
'''


class PowerReLU(nn.Module):
    """
    Implements simga(x)^(power)
    Applies a power of the rectified linear unit element-wise.

    NOTE: inplace may not be working.
    Can set inplace for inplace operation if desired.
    BUT I don't think it is working now.

    INPUT:
        x -- size (N,*) tensor where * is any number of additional
             dimensions
    OUTPUT:
        y -- size (N,*)
    """

    def __init__(self, inplace=False, power=3):
        super(PowerReLU, self).__init__()
        self.inplace = inplace
        self.power = power

    def forward(self, input):
        y = F.relu(input, inplace=self.inplace)
        return torch.pow(y, self.power)


class Block(nn.Module):
    """
    IMplementation of the block used in the Deep Ritz
    Paper

    Parameters:
    in_N  -- dimension of the input
    width -- number of nodes in the interior middle layer
    out_N -- dimension of the output
    phi   -- activation function used
    """

    def __init__(self, in_N, width, out_N, phi=PowerReLU()):
        super(Block, self).__init__()
        # create the necessary linear layers
        self.L1 = nn.Linear(in_N, width)
        self.L2 = nn.Linear(width, out_N)
        # choose appropriate activation function
        self.phi = nn.ReLU()

    def forward(self, x):
        return self.phi(self.L2(self.phi(self.L1(x)))) + x


class drrnn(nn.Module):
    """
    drrnn -- Deep Ritz Residual Neural Network

    Implements a network with the architecture used in the
    deep ritz method paper

    Parameters:
        in_N  -- input dimension
        out_N -- output dimension
        m     -- width of layers that form blocks
        depth -- number of blocks to be stacked
        phi   -- the activation function
    """

    def __init__(self, in_N, m, out_N, depth=4, phi=PowerReLU()):
        super(drrnn, self).__init__()
        # set parameters
        self.in_N = in_N
        self.m = m
        self.out_N = out_N
        self.depth = depth
        self.phi = nn.ReLU()

        # list for holding all the blocks
        self.stack = nn.ModuleList()

        # add first layer to list
        self.stack.append(nn.Linear(in_N, m))

        # add middle blocks to list
        for i in range(depth):
            self.stack.append(Block(m, m, m))

        # add output linear layer
        self.stack.append(nn.Linear(m, out_N))

    def forward(self, x):
        # first layer
        for i in range(len(self.stack)):
            x = self.stack[i](x)
        return x


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight)
        nn.init.constant_(m.bias, 0.0)

def get_interior_points(N=1000,d=10):
    """
    randomly sample N points from interior of [-1,1]^d
    """
    return torch.rand(N,d) * 2 - 1

def get_boundary_points(N=100):
    xb = torch.rand(2 * 10 * N, 10)
    for i in range(10):
        xb[2 * i * N: (2 * i + 1) * N, i] = 0.
        xb[(2 * i + 1) * N: (2 * i + 2) * N, i] = 1.

    return xb
def u(x):
    u = 0
    for i in range(5):
        u += x[:,2*i:2*i+1] * x[:,2*i+1:2*i+2]
    return u
def main():

    epochs = 20000
    in_N = 10
    m = 10
    out_N = 1

    print(torch.cuda.is_available())
    device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')
    model = drrnn(in_N, m, out_N).to(device)
    model.apply(weights_init)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    # x = torch.cat((xr, xb), dim=0)

    # if 2 < m:
    #     y = torch.zeros(x.shape[0], m - 2)
    #     x = torch.cat((x, y), dim=1)
    # # print(x.shape)
    best_loss, best_epoch = 1000, 0
    for epoch in range(epochs+1):

        # generate the data set
        xr = get_interior_points()
        xb = get_boundary_points()

        xr = xr.to(device)
        xb = xb.to(device)

        xr.requires_grad_()
        output_r = model(xr)
        output_b = model(xb)
        grads = autograd.grad(outputs=output_r, inputs=xr,
                              grad_outputs=torch.ones_like(output_r),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
        grads_sum = torch.sum(torch.pow(grads, 2), dim=1)
        u1 = 0.5 * grads_sum
        u1 = torch.mean(u1)
        u2 = torch.mean(torch.pow(output_b-u(xb), 2))
        loss = u1 + 20 * 500 * u2

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print('epoch:', epoch, 'loss:', loss.item(), 'loss_r:', u1.item(), 'loss_b:', (20* 500 * u2).item())
            if epoch > int(4 * epochs / 5):
                if torch.abs(loss) < best_loss:
                    best_loss = loss.item()
                    best_epoch = epoch
                    torch.save(model.state_dict(), 'new_best_high_dim1.mdl')
    print('best epoch:', best_epoch, 'best loss:', best_loss)


    # plot figure
    model.load_state_dict(torch.load('new_best_high_dim1.mdl'))
    print('load from ckpt!')

    with torch.no_grad():
        x = torch.rand(100000,10)
        u_exact = u(x)
        x = x.to(device)
        u_pred = model(x)
    err_l2 = torch.sqrt(torch.mean(torch.pow(u_pred-u_exact,2))) / torch.sqrt(torch.mean(torch.pow(u_exact,2)))
    print('L^2 relative error:', err_l2)


if __name__ == '__main__':
    main()
