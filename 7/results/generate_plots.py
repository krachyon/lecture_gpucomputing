import pandas as pd
from pylab import *
import matplotlib.cm as cm

dat=pd.read_csv('preliminary.csv')  

naive = dat[dat.name.map(lambda x: "naive" in x)]
shared = dat[dat.name.map(lambda x: "shared" in x)]


def threadcount(df, name):
    plt.figure()
    Ns = [128, 512, 896, 1664, 1920]
    for i,N in enumerate(Ns):
        curr = df[df.N == N]
        plot(curr.threads_per_block, N**2/curr.real_time, 'x', color = cm.nipy_spectral(i/(len(Ns))), label=f"problem size {N}")
    plt.xlabel("threads_per_block")
    plt.ylabel("FLOPS")
    plt.title(name)
    plt.legend()


threadcount(shared, 'shared')
threadcount(naive, 'naive')
plt.show()
