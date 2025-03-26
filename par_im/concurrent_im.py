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
    sampleset = np.zeros((samples, N, replicas))
    extvals, extvecs = torch.linalg.eigh(JExt)
    intvals, intvecs = torch.linalg.eigh(JInt)
    dom = extvecs[:, 0]
    # xt = torch.ones_like(dom)
    # torch.linalg.solve()
    # mat = JInt @ JExt
    # res = torch.linalg.solve(mat, x)
    # t1 = JInt.inverse() @ JExt @ torch.ones_like(x)
    breakpoint()
    samplecount = 0
    sumW = -J.sum()/2 * scale
    h = h.unsqueeze(-1)
    for i in tqdm(range(nsteps), disable=True):
        if samples > 0 and i >= burn_in_steps and i % sample_steps == 0:
            sampleset[samplecount] = x.cpu().numpy()
            samplecount += 1
            print(samplecount)
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
    breakpoint()
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
parser.add_argument('--tstop','-t', type=float, help='Total (unitless) simulation time', default=142)
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
    graph = os.environ['GSET']+'/set/G001'
    h, J, const = read_ising(graph, 1)
    x, samples, mean_gs, std_gs = simulate_concurrent_im(
                            h=h,
                            J=J,
                            beta0=1.0,
                            beta1=1.0,
                            tstop=args.tstop,
                            replicas=args.replicas,
                            samples=1000,
                            epoch=0.6451612903225806,
                            dt=args.dt,
                            nblocks=8,
                            size=None,
                            scale=1
                        )
    samples = samples.squeeze()
    eig = np.ones(h.shape[0])
    eig /= np.linalg.norm(eig)
    # samples /= np.linalg.norm(samples, axis=1)
    overlap = (samples.squeeze() @ eig) / np.linalg.norm(samples, axis=1)
    times = np.linspace(0, args.tstop, len(overlap))
    df = pd.DataFrame(np.expand_dims(times, -1), columns=['time'])
    df['overlap'] = overlap
    df.to_csv('g1_overlap.csv', index=False)