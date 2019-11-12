import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def clean_header(df):
    df.columns = [i.replace('#', '').strip() for i in df.columns]
    return df

def readData():
    return (
        clean_header(pd.read_csv('../results/memcopy.csv')),
        clean_header(pd.read_csv('../results/coalesced.csv')),
        clean_header(pd.read_csv('../results/stride_offset.csv'))
    )


def plotmemcopy(df):
    plt.figure()
    sizes = sorted(df['size'])
    plt.loglog(sizes, sorted(df['D2D(μs)']), label='D2D measured')
    plt.loglog(sizes, np.array(sizes)/(200*1e9*1e-6), label='minimum time with 200GB/s')

    plt.ylabel("Time to copy in μs")
    plt.xlabel("Size in bits")

    plt.legend()
    plt.savefig('mem.svg')


def plotcoalesced(df):

    fig = plt.figure(figsize=(10, 7))
    ax = fig.gca(projection='3d')
    ax.view_init(elev=16., azim=-160)
    x, y = np.meshgrid(np.unique(df.gDim), np.unique(df.bDim))
    surf = ax.plot_surface(np.log2(x), np.log2(y), np.array(df['bandwidth(GB/s)']).reshape(x.shape[0],x.shape[1]), cmap = cm.hot)
    fig.colorbar(surf, shrink=0.5)
    ax.set_xlabel(r'$\ln$ (Grid Dimension)', fontsize='x-large')
    ax.set_ylabel(r'$\ln$ (Block Dimension)', fontsize='x-large')
    ax.set_zlabel('Bandwidth (GB/s)', fontsize='x-large')
    ax.set_title('16MB')
    fig.tight_layout()
    fig.savefig('coalesced.svg')

def plotstride_offset(df):
    fig = plt.figure()
    for i in df.stride_offset.unique():
        stride = df[df.stride_offset == i]
        plt.semilogx(sorted(stride['size']), sorted(stride['bandwidth(GB/s)']), label=f'stride {i}')

    plt.xlabel('data size == gridDim * blockDim')
    plt.ylabel('Bandwidth in GB/s')
    plt.legend()
    plt.savefig(f'{df.type[0]}.svg')

if __name__=='__main__':
    mem, coalesced, so = readData()
    #plotmemcopy(mem)
    #plotcoalesced(coalesced)
    plotstride_offset(so[so.type == 'global-stride'])
    plt.show()