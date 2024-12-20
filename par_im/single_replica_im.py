import torch
import numpy as np
import pandas as pd
import os
from time import sleep
import re
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path
from concurrent_im import get_blocks, energy, read_ising

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def phasev(x: torch.Tensor, bits=1):
    if bits == 1:
        return x.cos().sign()
    # breakpoint()
    step = 2 / (1 << (bits))
    xstep = torch.ceil(x.cos() / step) * step
    return xstep

def ene_single(x: torch.Tensor, h: torch.Tensor, J: torch.Tensor):
    return -((h * x).sum(axis=0) + 0.5 * (x.T.matmul(J.matmul(x)))).diagonal()

def simulate_concurrent_kuramoto(J: torch.Tensor, 
                           h: torch.Tensor, 
                           scale0: float, 
                           scale1: float, 
                           tstop: float, 
                           dt: float,
                           epoch: float, 
                           samples: int = 0,
                           burn_in_time: int = 0,
                           nblocks: int = None, 
                           scale: float = 1,
                           size: int = None,
                           replicas: int = 1,
                           bits: int = 1,
                           k: float = 1):
    N = J.shape[0]
    blocks = get_blocks(N=N, nblocks=nblocks, maxsize=size)
    blocklist = [J[np.ix_(i, i)] for i in blocks]
    JInt: torch.Tensor = torch.block_diag(*blocklist)
    JExt = J - JInt
    x = (torch.rand((N, replicas), device=device) * 2 - 1).acos()
    nsteps = int(np.ceil(tstop / dt))
    sync_steps = int(np.ceil(epoch / dt))
    beta_steps = np.sqrt(dt)*np.linspace(scale0, scale1, nsteps)
    burn_in_steps = int(np.ceil(burn_in_time / dt))
    sample_steps = nsteps
    if samples > 0:
        sample_steps = int(np.floor((nsteps-burn_in_steps) / samples))
    sampleset = np.zeros((samples, N))
    samplecount = 0
    sumW = -J.sum()/2 * scale
    h.squeeze_().unsqueeze_(-1)
    for i in tqdm(range(nsteps), disable=True):
        # if samples > 0 and i >= burn_in_steps and i % sample_steps == 0:
        #     sampleset[(i-burn_in_steps) % sample_steps] = x.cpu().numpy()
        #     samplecount += 1
        if ((i % sync_steps) == 0):
            x_old = x.clone().detach()
            if bits > 0:
                x_old -= (x / (2 * np.pi)).floor() * 2 * np.pi
                step = 2 * np.pi / (1 << (bits))
                x_old = torch.round(x_old / step) * step
                
        
            
        repmat = x.unsqueeze(1).repeat((1, N, 1))
        diffmat = (k*(repmat - repmat.permute(1, 0, 2)).sin()).tanh()
        repmat2 = x_old.unsqueeze(1).repeat((1, N, 1))
        diffmat_ext = (k*(repmat - repmat2.permute(1, 0, 2)).sin()).tanh()
      
        gradient_int = torch.einsum('ij,ijk->jk', JInt, diffmat)
        gradient_ext = torch.einsum('ij,ijk->jk', JExt, diffmat_ext)

        gradient = gradient_int + gradient_ext - ((i % sync_steps) / sync_steps) * (2 * x).sin()
        x.add_(gradient, alpha=dt).add_(h, alpha=dt).add_(torch.randn_like(x), alpha=beta_steps[i])
        # if i % 10000 == 0:
        # if i % 1000 == 0:
        #     print((-ene_single(phasev(x), h, J) + sumW)/2)
        #     print(energy(x.sign(), h, J, scale, 0.0)/N)
        # print(ene_single(phasev(x), h, J))
    ene = (-ene_single(phasev(x), h, J) + sumW)/2
    # enevals = energy(phasev(x).unsqueeze(-1), h.unsqueeze(-1), J, scale, 0.0)/N
    return x, ene

def simulate_concurrent_brim(J: torch.Tensor, 
                           h: torch.Tensor, 
                           scale0: float, 
                           scale1: float, 
                           tstop: float, 
                           dt: float,
                           epoch: float, 
                           samples: int = 0,
                           burn_in_time: int = 0,
                           nblocks: int = None, 
                           scale: float = 1,
                           size: int = None,
                           replicas: int = 1,
                           bits: int = 1):
    N = J.shape[0]
    blocks = get_blocks(N=N, nblocks=nblocks, maxsize=size)
    blocklist = [J[np.ix_(i, i)] for i in blocks]
    JInt: torch.Tensor = torch.block_diag(*blocklist)
    JExt = J - JInt
    x = torch.rand((N, replicas), device=device) * 2 - 1
    nsteps = int(np.ceil(tstop / dt))
    sync_steps = int(np.ceil(epoch / dt))
    beta_steps = np.sqrt(dt)*np.linspace(scale0, scale1, nsteps)
    burn_in_steps = int(np.ceil(burn_in_time / dt))
    sample_steps = nsteps
    if samples > 0:
        sample_steps = int(np.floor((nsteps-burn_in_steps) / samples))
    sampleset = np.zeros((samples, N))
    samplecount = 0
    sumW = -J.sum()/2 * scale
    # breakpoint()
    h.squeeze_().unsqueeze_(-1)
    # x = torch.ones_like(x)
    for i in tqdm(range(nsteps), disable=True):
        # if samples > 0 and i >= burn_in_steps and i % sample_steps == 0:
        #     sampleset[(i-burn_in_steps) % sample_steps] = x.cpu().numpy()
        #     samplecount += 1
        if i % sync_steps == 0:
            if bits > 1:
                step = 1 / (1 << (bits-1))
                x_old = torch.ceil(x.abs() / step) * step * x.sign()
            elif bits == 1:
                x_old = x.sign()
            else:
                x_old = x.clone().detach()

        
        gradient_int = JInt.matmul(x)
        gradient_ext = JExt.matmul(x_old)
        gradient = gradient_int + gradient_ext
        x.add_(gradient, alpha=dt).add_(h, alpha=dt).add_(torch.randn_like(x), alpha=beta_steps[i]).clip_(-1, 1)
        # if i % 1000 == 0:
        #     print((-ene_single(torch.sign(x), h, J) + sumW)/2)
        # print(ene_single(phasev(x), h, J))
    ene = (-ene_single(torch.sign(x), h, J) + sumW)/2
    # enevals = energy(phasev(x).unsqueeze(-1), h.unsqueeze(-1), J, scale, 0.0)/N
    return x, ene

def simulate_kuramoto(J: torch.Tensor, 
                           h: torch.Tensor, 
                           scale0: float, 
                           scale1: float, 
                           tstop: float, 
                           dt: float,):
    N = J.shape[0]
    x = (torch.rand((N), device=device) * 2) - 1
    nsteps = int(np.ceil(tstop / dt))
    beta_steps = np.sqrt(dt)*np.linspace(scale0, scale1, nsteps)
    sumW = -J.sum()/2 
    for i in tqdm(range(nsteps), disable=True):
       
        gradient = (J * (x.sin().outer(x.cos()) - x.cos().outer(x.sin())).tanh()).sum(axis=0)
        gradient -=  (i / nsteps) * (2 * x).sin()
        # print(gradient, x.sin())
        x.add_(gradient, alpha=dt).add_(h, alpha=dt).add_(torch.randn_like(x), alpha=beta_steps[i])
        if i % 1000 == 0:
            print((-ene_single(torch.sign(x), h, J) + sumW)/2)
            # print(ene_single(phasev(x), h, J))
    x = phasev(x)
    ene = ene_single(x, h, J)
    # enevals = energy(phasev(x).unsqueeze(-1), h.unsqueeze(-1), J, scale, 0.0)/N
    return x, ene


parser = ArgumentParser()
parser.add_argument('--graph','-g', help='Path to graph edgelist', nargs='+')
parser.add_argument('--size','-s', type=int, help='Ising machine size (Default will assume problem size)', default=None)
parser.add_argument('--blocks','-b', type=int, help='Number of blocks to optimize (Default will assume problem size). NOTE: Do not specify both --size and --blocks', default=None, nargs='+')
parser.add_argument('--tstop','-t', type=float, help='Total (unitless) simulation time', default=100)
parser.add_argument('--dt', type=float, help='(Unitless) time step', default=1e-3)
parser.add_argument('--samples', type=int, help='Number of samples to collect', default=0)
parser.add_argument('--burn-in', type=float, help='Burn-in time', default=0)
parser.add_argument('--scale', type=float, help='Gradient scaling factor (e.g. RC time constant)', default=1)
parser.add_argument('--replicas', type=int, help='Number of spin replicas to simulate', default=1)
parser.add_argument('--epoch', type=float, help='Time between synchronization periods', default=1e-3)
parser.add_argument('--scale0', type=float, help='Starting noise coefficient (stddev)', default=1.5)
parser.add_argument('--scale1', type=float, help='Ending noise coefficient (stddev)', default=0.2)

if __name__ == '__main__':
    args = parser.parse_args()
    data = {
        'graph': [],
        'blocks': [],
        'epoch': [],
        'bandwidth': [],
        'solver': [],
        'cut': [],
        'k': [],
        'bits': []
    }
    tstop = args.tstop
    dt = 1e-3
    scale0 = 1.0
    scale1 = 0.0
    bits = 0
    for g in args.graph:
        h, J, const = read_ising(g, 1)
        for bandwidth in np.logspace(3, 1, 10, base=10)[-2:]:
            epoch = 1 / (31e4 * 50e-15 * bandwidth) * J.shape[0] / (2 << 30)
            for blocks in tqdm([1, 2, 4]):
                for k in [10]:
                    # for trials in range(10):
                    _, ene_km = simulate_concurrent_kuramoto(J=J, h=h, scale0=2.0, scale1=0.0, tstop=args.tstop, dt=args.dt, epoch=epoch, nblocks=blocks, scale=1, replicas=20, bits=bits, k=k)
                    # for ebrimval in ene_brim:
                    #     data['graph'].append(g)
                    #     data['blocks'].append(blocks)
                    #     data['epoch'].append(epoch)
                    #     data['bandwidth'].append(bandwidth)
                    #     data['cut'].append(ebrimval.item())
                    #     data['solver'].append('brim')
                    #     data['bits'].append(bits)
                    for ekmval in ene_km:
                        data['graph'].append(g)
                        data['blocks'].append(blocks)
                        data['epoch'].append(epoch)
                        data['bandwidth'].append(bandwidth)
                        data['cut'].append(ekmval.item())
                        data['solver'].append('kuramoto')
                        data['bits'].append(bits)
                        data['k'].append(k)
                df = pd.DataFrame(data)
                df.to_csv("kuramoto_linear_comparison_quantized_k10_g18_10_2.csv", index=False)

                # simulate_kuramoto(J, h, 2.0, 0.5, 200, 1e-3)