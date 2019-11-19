#! /usr/bin/python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
plt.ioff()
def clean_header(df):
    df.columns = [i.replace('#', '').strip() for i in df.columns]
    return df

def readData():
    return (
        clean_header(pd.read_csv('./g2s_1b.csv')),
        clean_header(pd.read_csv('./s2g_1b.csv')),
        clean_header(pd.read_csv('./g2s.csv')),
        clean_header(pd.read_csv('./s2g.csv')),
        clean_header(pd.read_csv('./s2r.csv')),
        clean_header(pd.read_csv('./r2s.csv')),
        clean_header(pd.read_csv('./s2r_c.csv')),
    )


def global2shared_1b(df):
    plt.figure(figsize=(10,10))
    plt.title("Bandwitdh for global2shared for 1 thread block")
    sizes = sorted(pd.unique(df['size']))
    threads = sorted(pd.unique(df['bDim']))
    for thread in threads:
        data=df.loc[df['bDim'] == thread]
        plt.loglog([x / 1000 for x in sizes], data['bw in GB/s'], label='%d threads' % thread)

    plt.ylabel("Bandwith in GB/s")
    plt.xlabel("size in kB")

    plt.legend(loc=2)
    plt.savefig('g2s_1b.svg')

def shared2global_1b(df):
    plt.figure(figsize=(10,10))
    plt.title("Bandwitdh for shared2global for 1 thread block")
    sizes = sorted(pd.unique(df['size']))
    threads = sorted(pd.unique(df['bDim']))
    for thread in threads:
        data=df.loc[df['bDim'] == thread]
        plt.loglog([x / 1000 for x in sizes], data['bw in GB/s'], label='%d threads' % thread)

    plt.ylabel("Bandwith in GB/s")
    plt.xlabel("size in kB")

    plt.legend(loc=2)
    plt.savefig('s2g_1b.svg')

def global2shared(df):
    plt.figure(figsize=(10,10))
    plt.title("Bandwitdh for global2shared for 10 kB data size")
    blocks = sorted(pd.unique(df['gDim']))
    threads = sorted(pd.unique(df['bDim']))
    for thread in threads:
        data=df.loc[df['bDim'] == thread]
        plt.plot(blocks,  [x/y for x, y in zip(data['bw in GB/s'],blocks)], label='%d threads' % thread)

    plt.ylabel("Bandwith in GB/s / block count")
    plt.xlabel("block count")

    plt.legend(loc=4)
    plt.yscale("log")
    plt.savefig('g2s.svg')

def shared2global(df):
    plt.figure(figsize=(10,10))
    plt.title("Bandwitdh for shared2global for 10 kB data size")
    blocks = sorted(pd.unique(df['gDim']))
    threads = sorted(pd.unique(df['bDim']))
    for thread in threads:
        data=df.loc[df['bDim'] == thread]
        plt.plot(blocks, [x/y for x, y in zip(data['bw in GB/s'],blocks)], label='%d threads' % thread)

    plt.ylabel("Bandwith in GB/s / block count")
    plt.xlabel("block count")

    plt.legend(loc=4)
    plt.yscale("log")
    plt.savefig('s2g.svg')

def shared2register(df):
    blocks = sorted(pd.unique(df['gDim']))
    threads = sorted(pd.unique(df['bDim']))
    sizes = sorted(pd.unique(df['size']))
    for size in sizes:
        plt.figure(figsize=(10,10))
        plt.title("Bandwitdh for shared2register for %d kB" % size)
        data=df.loc[df['size'] == size]
        for thread in threads:
            data1=data.loc[data['bDim'] == thread]
            plt.plot(blocks, [x/y for x, y in zip(data1['bw in GB/s'],blocks)], label='%d threads' % thread)

        plt.ylabel("Bandwith in GB/s / block count")
        plt.xlabel("block count")

        plt.legend(loc=4)
        plt.yscale("log")
        plt.savefig('s2r%d.svg' %size)

def register2shared(df):
    blocks = sorted(pd.unique(df['gDim']))
    threads = sorted(pd.unique(df['bDim']))
    sizes = sorted(pd.unique(df['size']))
    for size in sizes:
        plt.figure(figsize=(10,10))
        plt.title("Bandwitdh for register2shared for %d kB" % size)
        data=df.loc[df['size'] == size]
        for thread in threads:
            data1=data.loc[data['bDim'] == thread]
            plt.plot(blocks, [x/y for x, y in zip(data1['bw in GB/s'],blocks)], label='%d threads' % thread)

        plt.ylabel("Bandwith in GB/s / block count")
        plt.xlabel("block count")

        plt.legend(loc=4)
        plt.yscale("log")
        plt.savefig('r2s%d.svg' %size)

def shared2register_conf(df):
    threads = sorted(pd.unique(df['bDim']))
    strides = sorted(pd.unique(df['stride']))
    plt.figure(figsize=(10,10))
    plt.title("Computation times for bank conflicts")
    for thread in threads:
        data=df.loc[df['bDim'] == thread]
        plt.plot(strides, data['clock'], label='%d threads' % thread)

    plt.ylabel("clock count")
    plt.xlabel("stride")

    plt.legend(loc=2)
    plt.savefig('s2r_c.svg')


if __name__=='__main__':
    g2s_1b, s2g_1b, g2s, s2g, s2r, r2s, s2r_c = readData()
    global2shared_1b(g2s_1b)
    shared2global_1b(s2g_1b)
    global2shared(g2s)
    shared2global(s2g)
    shared2register(s2r)
    register2shared(r2s)
    shared2register_conf(s2r_c)	

   # plt.show()
