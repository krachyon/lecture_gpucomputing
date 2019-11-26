#! /usr/bin/python
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.ioff()


def plot_matrix():    
    caches = [np.sqrt(32*1024/4), np.sqrt(256 * 1024/4), np.sqrt(5*1024*1024/4)]

    dat_cpu = pd.read_csv('../results/mmul_cpu.csv', comment='#', header=None)
    dat_cpu.columns = ['method','iters', 'N', 'time']

    dat_gpu = pd.read_csv('../results/mmul_cuda.csv', comment='#', header=None)
    dat_gpu.columns = ['method', 'iters', 'N', 'threadsize', 'memtime', 'time']

    

    fig = plt.figure(figsize=(11, 8))

    for method in ['cpu', 'eigen']:
        df = dat_cpu[dat_cpu.method == method]

        plt.plot(df.N, 2*df.N**3/(df.time/1e9), 'x', ms=6, alpha=0.6,
                label=method)
    
    for method in ['cuda_shared', 'cuda_naive']:
    
        df = dat_gpu[(dat_gpu.method == method) & (dat_gpu.threadsize==8)]

        plt.plot(df.N, 2*df.N**3/(df.time/1e9), 'x', ms=6, alpha=0.6,
                label=method)
    

    plt.xlabel('Matrix<float> size NxN')
    plt.ylabel('Flops/s')
    plt.axvline(caches[0], label='L1 cache', color='yellow')
    plt.axvline(caches[1], label='L2 cache', color='orange')
    plt.axvline(caches[2], label='half L3 cache', color='tab:red')
    plt.legend()
    plt.xlim(0, 2500)
    plt.tight_layout()
    plt.savefig(f'cudavcpu_matrix_flops.pdf')

    fig = plt.figure(figsize=(11, 8))


    for threadsize in dat_gpu.threadsize.unique():
        cindex = np.log2(threadsize)/np.log2(32)

        method,color = 'cuda_shared', cm.rainbow(cindex)
        df = dat_gpu[(dat_gpu.method == method) & (dat_gpu.threadsize==threadsize)]
        df.dropna()

        if len(df) > 0:
            plt.plot(df.N, 2*df.N**3/(df.time/1e9), 'x', ms=6, alpha=0.6,
                label=f'{method} threads ${threadsize}^2$', color=color)
    plt.xlabel('Matrix<float> size NxN')
    plt.ylabel('Flops/s')
    plt.legend()
    plt.xlim(0, 2500)
    plt.tight_layout()
    plt.savefig(f'shared_sizes.pdf')
    
    fig = plt.figure(figsize=(11, 8))
    for threadsize in dat_gpu.threadsize.unique():
        cindex = np.log2(threadsize)/np.log2(32)
        method,color = 'cuda_naive', cm.rainbow(cindex)
        df = dat_gpu[(dat_gpu.method == method) & (dat_gpu.threadsize==threadsize)]
        df.dropna()

        if len(df) > 0:
            plt.plot(df.N, 2*df.N**3/(df.time/1e9), 'x', ms=6, alpha=0.6,
                label=f'{method} threads ${threadsize}^2$', color=color)
    
    plt.xlabel('Matrix<float> size NxN')
    plt.ylabel('Flops/s')
    plt.legend()
    plt.xlim(0, 2500)
    plt.tight_layout()
    plt.savefig(f'naive_sizes.pdf')

    fig = plt.figure(figsize=(11, 8))
    threadsize,method = 8,'cuda_shared'
    df = dat_gpu[(dat_gpu.method == method) & (dat_gpu.threadsize==threadsize)]
    df=df[(df.N > 2)]
    df.dropna()

    if len(df) > 0:
        plt.plot(df.N,(df.memtime/df.time), 'x', ms=6, alpha=0.6,
            label=f'memtime {method} threads ${threadsize}^2$')
        plt.plot(df.N, ((df.time-df.memtime)/df.time), 'x', ms=6, alpha=0.6,
            label=f'comptime {method} threads ${threadsize}^2$')
    plt.xlabel('Matrix<float> size NxN')
    plt.ylabel('fraction of the runtime')
    plt.legend()
    plt.xlim(0, 2500)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig(f'mem_comp_splitting.pdf')


def clean_header(df):
    df.columns = [i.replace('#', '').strip() for i in df.columns]
    return df


if __name__ == '__main__':
    plot_matrix()
    plt.show()
