import pandas as pd
from pylab import *
import matplotlib.cm as cm

dat=pd.read_csv('preliminary.csv')  

naive = dat[dat.name.map(lambda x: "naive" in x)]
shared = dat[dat.name.map(lambda x: "shared" in x)]
unaligned = dat[dat.name.map(lambda x: "unaligned" in x)]


def threadcount(df, name):
    plt.figure(figsize=(10,8))
    Ns = [128, 512, 896, 1664, 1920]
    for i,N in enumerate(Ns):
        curr = df[df.N == N]
        plot(curr.threads_per_block, N**2/(curr.real_time/1000), 'x', color = cm.nipy_spectral(i/(len(Ns))), label=f"problem size {N}")
    plt.xlabel("threads_per_block")
    plt.ylabel("FLOPS")
    plt.title(name)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'threadcount_{name}.pdf')

def heatmap(df, name):
    plt.figure(figsize=(10,8))

    piv = pd.pivot_table(df, index='N',columns='threads_per_block',values='real_time', dropna=True, aggfunc=mean) 
    piv /=1000  # milliseconds to seconds
    piv = 1./ piv.divide(piv.index**2, axis='rows')
    
    plt.pcolormesh(log(piv), cmap=cm.terrain, vmin=16, vmax=21)
    plt.yticks(range(len(piv.index)),piv.index)
    plt.xticks(piv.columns[::50])
    plt.xlabel('threads per block')
    plt.ylabel('problem size')
    plt.title(f'$\\log_{{10}}$(Flops) {name}')
    colorbar()
    plt.tight_layout()
    plt.savefig(f'heatmap_{name}.pdf')



def alignment(full_data):
    plt.figure(figsize=(10,8))
    naive = full_data[dat.name.map(lambda x: "naive" in x) & (full_data.N == 768)]
    unaligned = full_data[dat.name.map(lambda x: "unaligned" in x) & (full_data.N == 768)]

    plt.plot(naive.threads_per_block, naive.N**2/(naive.real_time/1000), 'x', alpha=0.7, label='align(32)')
    plt.plot(unaligned.threads_per_block, unaligned.N**2/(unaligned.real_time/1000), 'x', alpha=0.7, label='unaligned')
    plt.title('impact of alignment on performance, N=768')
    plt.xlabel('threads per block')
    plt.ylabel('Flops')
    plt.legend()
    plt.tight_layout()
    plt.savefig('alignment.pdf')



def nvsflops(naive_orig, shared_orig):
    plt.figure(figsize=(10,8))
    naive = naive_orig.copy().groupby('N').min()
    shared = shared_orig.copy().groupby('N').min()

    plt.plot(naive.index, naive.index**2/(naive.real_time/1000), label='naive')
    plt.plot(shared.index, shared.index**2/(shared.real_time/1000), label='shared')
    plt.xlabel('problem size')
    plt.ylabel('Flops for optimum kernel configuration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nvsflops1.pdf')


    naive = naive_orig.copy().groupby('N').mean()
    shared = shared_orig.copy().groupby('N').mean()
    
    plt.figure(figsize=(10,8))
    plt.plot(naive.index, naive.index**2/(naive.real_time/1000), label='naive')
    plt.plot(shared.index, shared.index**2/(shared.real_time/1000), label='shared')
    plt.xlabel('problem size')
    plt.ylabel('Flops for average kernel configuration')
    plt.legend()
    plt.tight_layout()
    plt.savefig('nvsflops2.pdf')



threadcount(shared, 'shared')
threadcount(naive, 'naive')
heatmap(shared, 'shared')
heatmap(naive, 'naive')
#heatmap(unaligned, 'unaligned')
alignment(dat)
nvsflops(naive, shared)

plt.show()
