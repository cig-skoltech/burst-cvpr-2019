import torch
import torch
from torch import nn
import numpy as np

idx = 0


class MMNet(torch.nn.Module):
    def __init__(self, model, max_iter=10, sigma_max=2, sigma_min=1):
        r""" MMNet implements the defined optimaztion scheme which unrolls a defined number of iterations.
        """
        super(MMNet, self).__init__()
        self.model = model  # the denoising model
        self.max_iter = max_iter  # number of maximum iterations
        # continuation scheme is defined by alpha since it scales the estimated standard deviation
        self.alpha = nn.Parameter(torch.Tensor(np.linspace(np.log(sigma_max), np.log(sigma_min), max_iter)))
        iterations = np.arange(self.max_iter)
        iterations[0] = 1
        iterations = np.log(iterations / (iterations + 3))
        w = nn.Parameter(torch.Tensor(iterations))  # Extrapolation parameters, initialized as in
        # Boyd Proximal Algorithms
        self.w = w

    def forward(self, xcur, xpre, p, k):

        r""" Implements a single iteration with index k. The variable xcur is the current solution, xpre is the previous
        solution and p is the object of the relevant problem Class.
        """

        if k > 0:
            wk = self.w[k]
            yk = xcur + torch.exp(wk) * (xcur - xpre)  # extrapolation step
        else:
            yk = xcur

        xpre = xcur
        net_input = yk - p.energy_grad(yk)
        noise_sigma = p.L
        xcur = (net_input - self.model(net_input, noise_sigma, self.alpha[k]))  # residual approach of model
        xcur = xcur.clamp(0, 255)  # clamp to ensure correctness of representation
        return xcur, xpre

    def forward_all_iter(self, p, init, noise_estimation, max_iter=None):
        r""" Implements a certain number of iterations. If max_iter is None then all iterations are run, in every other
        case we unroll up to max_iter iterations.
        """
        if max_iter is None:
            max_iter = self.max_iter

        xcur = p.y

        if init:  # initialize optimization scheme
            xcur = p.initialize()

        if noise_estimation:  # estimate noise standard deviation
            p.estimate_noise()

        xpre = 0
        for i in range(max_iter):
            xcur, xpre = self.forward(xcur, xpre, p, i)

        return xcur


class TBPTT(torch.nn.Module):
    def __init__(self, model, loss_module, k1, k2, optimizer, max_iter=20, clip_grad=0.25):
        r""" Implementation of Truncated Backpropagation through Time for iterative optimization schemes.
        Code is based on https://discuss.pytorch.org/t/implementing-truncated-backpropagation-through-time/15500/2
        """
        super(TBPTT, self).__init__()
        self.model = model  # denoising model
        self.max_iter = max_iter  # number of maximum iterations to unroll
        self.loss_module = loss_module  # loss criterion to be used for training
        self.k1 = k1  # number of forward iterations to unroll
        self.k2 = k2  # number of iterations to backpropagate into. Counting starts from last iteration.
        self.retain_graph = k1 <= k2
        self.clip_grad = clip_grad
        self.optimizer = optimizer  # optimizer to be used for gradient updates

    def train(self, p, target, init, noise_estimation=False):
        xcur = p.y
        if init:  # initialize optimization scheme
            xcur = p.initialize()

        if noise_estimation:  # estimate noise standard deviation
            p.estimate_noise()

        xpre = 0
        states = [(None, xcur)]
        for i in range(self.max_iter):
            state = states[-1][1].detach()
            state.requires_grad = True

            xcur, xpre = self.model(state, xpre, p, i)
            new_state = xcur
            states.append((state, new_state))

            while len(states) > self.k2:
                # Delete stuff that is too old
                del states[0]

            if (i + 1) % self.k1 == 0:
                loss = self.loss_module(xcur, target)
                if i + 1 != self.max_iter:
                    loss = loss * 0.5
                self.optimizer.zero_grad()
                # backprop last module (keep graph only if they ever overlap)
                loss.backward(retain_graph=self.retain_graph)
                for i in range(self.k2 - 1):
                    # if we get all the way back to the "init_state", stop
                    if states[-i - 2][0] is None:
                        break
                    curr_grad = states[-i - 1][0].grad
                    states[-i - 2][1].backward(curr_grad, retain_graph=self.retain_graph)
                # Clip gradient
                if self.clip_grad is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
                self.optimizer.step()
        self.model.zero_grad()
        return xcur


if __name__ == '__main__':
    import numpy as np
    from problems import *

    np.random.seed(42)
    torch.manual_seed(42)
    from ResDNet import *
    import utilities

    # compile and load pre-trained model
    model = ResDNet(BasicBlock, 3, weightnorm=True)
    size = [2, 3, 100, 100]
    M = np.random.randn(*size)
    y = np.random.randn(*size)
    p = Demosaic(torch.FloatTensor(y), torch.FloatTensor(M), True)
    p.cuda_()
    target = np.random.randn(*size)
    criterion = nn.MSELoss()
    max_iter = 2
    mmnet = MMNet(model, max_iter=max_iter)
    mmnet = mmnet.cuda()
    optimizer = torch.optim.Adam(mmnet.parameters(), lr=1e-2)
    runner = TBPTT(mmnet, criterion, 1, 1, optimizer, max_iter=max_iter)
    print(criterion(Variable(torch.Tensor(y)), Variable(torch.Tensor(target))).item())
    for i in range(200):
        idx = 0
        out = runner.train(p, torch.Tensor(target).cuda(), init=True)
        print(criterion(out, torch.Tensor(target).cuda()).item())
