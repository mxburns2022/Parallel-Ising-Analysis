import torch
import numpy as np
import pandas as pd
import os
import re
from argparse import ArgumentParser
from tqdm import tqdm
from pathlib import Path

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'



def get_blocks(N: int, maxsize: int, nblocks: int):
    if nblocks is None:
        assert maxsize is not None
        nblocks = int(np.ceil(N / maxsize))
    return np.array_split(np.arange(N), nblocks)


def simulate_concurrent_im(J: torch.Tensor, 
                           h: torch.Tensor, 
                           beta0: float, 
                           beta1: float, 
                           tstop: float, 
                           dt: float,
                           epoch: float, 
                           replicas: int = 1,
                           samples: int = 0,
                           burn_in_time: int = 0,
                           nblocks: int = None, 
                           const: float = 1,
                           scale: float = 1,
                           size: int = None):
    N = J.shape[0]
    blocks = get_blocks(N=N, nblocks=nblocks, maxsize=size)
    blocklist = [J[np.ix_(i, i)] for i in blocks]
    JInt: torch.Tensor = torch.block_diag(*blocklist)
    JExt = J - JInt
    x = torch.randn((N, replicas), device=device)
    nsteps = int(np.ceil(tstop / dt))
    sync_steps = int(np.ceil(epoch / dt))
    beta_steps = np.sqrt(2*dt)*np.sqrt(1/(scale*np.linspace(beta0, beta1, nsteps)))
    burn_in_steps = int(np.ceil(burn_in_time / dt))
    sample_steps = nsteps
    if samples > 0:
        sample_steps = int(np.floor((nsteps-burn_in_steps) / samples))
    sampleset = np.zeros((samples, N))
    samplecount = 0
    sumW = -J.sum()/2 * scale
    h = h.unsqueeze(-1)
    for i in tqdm(range(nsteps), disable=True):
        if samples > 0 and i >= burn_in_steps and i % sample_steps == 0:
            sampleset[(i-burn_in_steps) % sample_steps] = x.cpu().numpy()
            samplecount += 1
        if i % sync_steps == 0:
            x_old = x.clone().detach()
        gradient = JInt.mm(x) + JExt.mm(x_old)
        x.add_(gradient, alpha=dt).add_(h, alpha=dt).add_(torch.randn_like(x), alpha=beta_steps[i]).clip_(-1, 1)
        # if i % 10000 == 0:
        #     print(energy(x.sign(), h, J, scale, 0.0)/N)
    enevals = energy(x.sign(), h, J, scale, 0.0)/N
    return x, sampleset, enevals.mean().item(), enevals.std().item()



def simulate_concurrent_im_sparse(J: torch.Tensor, 
                           h: torch.Tensor, 
                           beta0: float, 
                           beta1: float, 
                           tstop: float, 
                           dt: float,
                           epoch: float, 
                           replicas: int = 1,
                           samples: int = 0,
                           burn_in_time: int = 0,
                           nblocks: int = None, 
                           const: float = 1,
                           scale: float = 1,
                           size: int = None):
    N = J.shape[0]
    blocks = get_blocks(N=N, nblocks=nblocks, maxsize=size)
    blocklist = [J[np.ix_(i, i)] for i in blocks]
    JInt: torch.Tensor = torch.block_diag(*blocklist)
    JExt = J - JInt
    x = torch.randn((N, replicas), device=device)
    nsteps = int(np.ceil(tstop / dt))
    sync_steps = int(np.ceil(epoch / dt))
    beta_steps = np.sqrt(2*dt)*np.linspace(beta0, beta1, nsteps)
    burn_in_steps = int(np.ceil(burn_in_time / dt))
    sample_steps = nsteps
    if samples > 0:
        sample_steps = int(np.floor((nsteps-burn_in_steps) / samples))
    sampleset = np.zeros((samples, N))
    samplecount = 0
    sumW = -J.sum()/2 * scale
    h = h.unsqueeze(-1)
    for i in tqdm(range(nsteps), disable=True):
        if samples > 0 and i >= burn_in_steps and i % sample_steps == 0:
            sampleset[(i-burn_in_steps) % sample_steps] = x.cpu().numpy()
            samplecount += 1
        if i % sync_steps == 0:
            x_old = x.clone().detach()
        gradient = JInt.mm(x) + JExt.mm(x_old)
        x.add_(gradient, alpha=dt).add_(h, alpha=dt).add_(torch.randn_like(x), alpha=beta_steps[i]).clip_(-1, 1)
        # if i % 10000 == 0:
        #     print(energy(x.sign(), h, J, scale, 0.0)/N)
    enevals = energy(x.sign(), h, J, scale, 0.0)/N
    return x, sampleset, enevals.mean().item(), enevals.std().item()

def simulate_full_im(J: torch.Tensor, 
                           h: torch.Tensor, 
                           beta0: float, 
                           beta1: float, 
                           tstop: float, 
                           dt: float,
                           epoch: float, 
                           replicas: int = 1,
                           samples: int = 0,
                           burn_in_time: int = 0,
                           const: float = 1,
                           scale: float = 1):
    N = J.shape[0]
    x = torch.randn((N, replicas), device=device)
    nsteps = int(np.ceil(tstop / dt))
    beta_steps = np.sqrt(2*dt)*scale*np.linspace(beta0, beta1, nsteps)
    burn_in_steps = int(np.ceil(burn_in_time / dt))
    sample_steps = nsteps
    if samples > 0:
        sample_steps = int(np.floor((nsteps-burn_in_steps) / samples))
    sampleset = np.zeros((samples, N))
    samplecount = 0
    sumW = -J.sum()/2 * scale
    h = h.unsqueeze(-1)
    for i in tqdm(range(nsteps), disable=True):
        if samples > 0 and i >= burn_in_steps and i % sample_steps == 0:
            sampleset[(i-burn_in_steps) % sample_steps] = x.cpu().numpy()
            samplecount += 1
        gradient = J.mm(x)
        x.add_(gradient, alpha=dt).add_(h, alpha=dt).add_(torch.randn_like(x), alpha=beta_steps[i]).clip_(-1, 1)
    enevals = energy(x.sign(), h, J, scale, 0.0)/N
    return x, sampleset, enevals.mean().item(), enevals.std().item()

def to_numeric(val: str):
    try:
        return int(val)
    except ValueError:
        return float(val)


def read_ising(path: str, scale: float):
    with open(path) as infile:
        if '.ising' in path:
            nodes, nlin, edges = map(int, infile.readline().strip().split())
            # const = float(infile.readline())
            const=0
            offset = 0
        else:
            offset = 0 if '.gset' in path else 1
            nodes, edges = map(int, infile.readline().strip().split())
            const = 0
            nlin = 0
        h = torch.zeros(nodes, device=device)
        J = torch.zeros((nodes, nodes), device=device)
        for i in range(nlin):
            args = infile.readline().strip().split()
            u, val = map(to_numeric, args)
            h[u-offset] = val
        for i in range(edges):
            args = infile.readline().strip().split()
            u, v, val = map(to_numeric, args)
            J[u-offset][v-offset] = -val
            J[v-offset][u-offset] = -val
        h /= scale
        J /= scale
        # h /= J.abs().max()
        # J /= J.abs().max()
        return h, J, const
        
def energy(x: torch.Tensor, 
           h: torch.Tensor, 
           J: torch.Tensor, 
           scale: float, 
           const: float):
    return ((-0.5 * x.T.mm(J).mm(x) - x.T.mm(h)) * scale - const).diag()



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
parser.add_argument('--beta0', type=float, help='Starting inverse temperature', default=0.2)
parser.add_argument('--beta1', type=float, help='Ending inverse temperature', default=2.0)

if __name__ == '__main__':
    args = parser.parse_args()
    for blocks in args.blocks:
        logfile = f'rng_tsp_block_{blocks}_log_2.csv'
        with open(logfile, 'w') as log:
            log.write(f'prob,blocks,tstop,beta0,beta1,epoch,epoch_real,syncs(MHz),beta0,beta1,mean_gs,std_gs,spin\n')
            for graph in tqdm(args.graph):
                try:
                    cities, ind = map(int, re.findall(r'TSP_(\d+)_(\d+)\.ising', graph)[0])
                except IndexError:
                    continue
                
                if cities == 20:
                    beta1 = 22
                if cities == 25:
                    beta1 = 28
                if cities == 30:
                    beta1 = 37
                if cities == 35:
                    beta1 = 46
                
                # print()
                if ind >= 20:
                    continue
                h, J, const = read_ising(graph, args.scale)
                # print(J.mean(), J.)
                # args.scale 
                for syncs in tqdm(np.logspace(3, 0, 40, base=10)):
                    epoch_real = 1 / (syncs * 1e6)
                    epoch = epoch_real / (50e-15 * 31e4)
                    log.flush()
                    if (args.size is not None and args.size < J.shape[0]) or blocks > 1:
                        x, samples, mean_gs, std_gs = simulate_concurrent_im(
                            h=h,
                            J=J,
                            beta0=args.beta0,
                            beta1=beta1,
                            tstop=args.tstop,
                            replicas=args.replicas,
                            epoch=epoch,
                            dt=args.dt,
                            nblocks=blocks,
                            size=args.size,
                            scale=args.scale
                        )
                    else:
                        x, samples, mean_gs, std_gs = simulate_full_im(
                            h=h,
                            J=J,
                            beta0=args.beta0,
                            beta1=beta1,
                            tstop=args.tstop,
                            replicas=args.replicas,
                            epoch=epoch,
                            dt=args.dt,
                            scale=args.scale
                        )
                        
                    for xi in x.T:
                        log.write(f'{graph},{blocks},{args.tstop},{epoch},{epoch_real},{syncs},{args.beta0},{beta1},{mean_gs},{std_gs},{":".join([str(i.item()) for i in xi.sign()])}\n')
