#! /usr/bin/python
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

plt.ioff()


def plot_matrix(dataname):    
    caches = [np.sqrt(32*1024/4), np.sqrt(256 * 1024/4), np.sqrt(5*1024*1024/4)]

    dat = pd.read_csv(dataname, comment='#', header=None)
    dat.columns = ['method','iters', 'N', 'time']

    fig = plt.figure(figsize=(11, 8))

    for method in ['cpu', 'eigen', 'cuda']:
        df = dat[dat.method == method]

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
    plt.savefig(f'cudavcpu_matrix_flops.svg')


def clean_header(df):
    df.columns = [i.replace('#', '').strip() for i in df.columns]
    return df


if __name__ == '__main__':
    plot_matrix('../results/mmul.csv')
    plt.show()
