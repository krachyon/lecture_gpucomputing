import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from cycler import cycler

def calc_bandwidth(df):
    """given a data frame with expected format, add bandwidth in GiB/s to df"""
    elem_size = 4
    seconds_per_ns = 1e9
    GiB = 1024*1024*1024

    df["bw_exec"] = df.N_iter * df.N_elem * elem_size / df.t_exec * seconds_per_ns / GiB
    df["bw_full"] = df.N_iter * df.N_elem * elem_size / df.t_tot * seconds_per_ns / GiB


dat=pd.read_csv('result_final.csv',comment='#', header=None)
dat.columns="N_elem,N_iter,N_block,dtype,method,t_tot,t_copy,t_exec,t_backcopy".split(',')

niter = pd.unique(dat.N_iter)
calc_bandwidth(dat)

std=dat[dat.method=='std::accumulate']
cpu=dat[dat.method=='cpu']
thrust=dat[dat.method=='thrust']
naive=dat[(dat.method=='cuda_naive')&(dat.dtype=='float')]
shared=dat[(dat.method=='cuda_shared')&(dat.dtype=='float')]
optim=dat[(dat.method=='cuda_optim')&(dat.dtype=='float')]

def full_bw():
    plt.figure(figsize=(10,8))
    plt.loglog(naive.N_elem, naive.bw_full, 'x', label='cuda_global', alpha=0.7)
    plt.loglog(shared.N_elem, shared.bw_full, 'x', label='cuda_shared', alpha=0.7)
    plt.loglog(optim.N_elem, optim.bw_full, 'x', label='cuda_optim', alpha=0.7)
    plt.loglog(thrust.N_elem, thrust.bw_full, 'x', label='thrust::accumulate', alpha=0.7)
    plt.loglog(cpu.N_elem, cpu.bw_full,'x', label='cpu own', alpha=0.7)
    plt.loglog(std.N_elem, std.bw_full,'x', label='std::accumulate', alpha=0.7)


    plt.title("throughput with copy time")
    plt.xlabel("number of elements")
    plt.ylabel("bandwidth [GiB/s]")
    plt.legend()
    plt.savefig('full_bw.pdf')


def exec_bw():
    plt.figure(figsize=(10,8))
    plt.loglog(naive.N_elem, naive.bw_exec, 'x', label='cuda_global', alpha=0.7)
    plt.loglog(shared.N_elem, shared.bw_exec, 'x', label='cuda_shared', alpha=0.7)
    plt.loglog(optim.N_elem, optim.bw_exec, 'x', label='cuda_optim', alpha=0.7)

    plt.loglog(thrust.N_elem, thrust.bw_exec,'x', label='thrust::accumulate', alpha=0.7)
    # Note that this is still bw_full as copy time makes no sense for cpu
    plt.loglog(cpu.N_elem, cpu.bw_full,'x', label='cpu own', alpha=0.7)
    plt.loglog(std.N_elem, std.bw_full,'x', label='std::accumulate', alpha=0.7)

    plt.title("throughput excluding copy time")
    plt.xlabel("number of elements")
    plt.ylabel("bandwidth [GiB/s]")
    plt.legend()
    plt.savefig('exec_bw.pdf')

def blocksize():
    plt.rc('axes', prop_cycle=cycler(color=[cm.nipy_spectral(i/7) for i in range(7)]))
    plt.figure(figsize=(10,8))
    for bs in sorted(pd.unique(dat.N_block)):
        naive_n = dat[(dat.method == 'cuda_naive') & (dat.dtype == 'float') & (dat.N_block == bs)]
        plt.loglog(naive_n.N_elem, naive_n.bw_exec, 'x', label=f'naive bs={int(bs)}', alpha=0.7)

    plt.title("blocksize dependent throughput, naive")
    plt.xlabel("number of elements")
    plt.ylabel("bandwidth [GiB/s]")
    plt.legend()
    plt.savefig('naive_bs.pdf')

    plt.figure(figsize=(10,8))
    for bs in sorted(pd.unique(dat.N_block)):
        shared_n = dat[(dat.method == 'cuda_shared') & (dat.dtype == 'float') & (dat.N_block == bs)]
        plt.loglog(shared_n.N_elem, shared_n.bw_exec, 'x', label=f'shared bs={int(bs)}', alpha=0.7)

    plt.xlabel("blocksize dependent throughput, shared")
    plt.ylabel("bandwidth [GiB/s]")
    plt.legend()
    plt.savefig('shared_bs.pdf')




if __name__=='__main__':
    full_bw()
    exec_bw()
    blocksize()

    plt.show()
