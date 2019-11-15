import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_matrix(dataname,cpuname,caches):
    fig = plt.figure(figsize=(11, 8))
    dat = pd.read_csv(dataname,comment='#', header=None)
    dat.columns = ['iters', 'N', 'time']

    dat = dat.sort_values(by=['N']) 
    plt.plot(dat.N, dat.time/(dat.size**2),'x',ms=6,alpha=0.6, label=cpuname, color='tab:red' if 'Ryzen' in cpuname else 'tab:blue')

    plt.axvline(caches[1], label='L2 Cache', color='orange')
    plt.axvline(caches[2], label='half L3 Cache',color='blue' if 'Ryzen' in cpuname else 'tab:red')

    plt.xlabel('matrix<float> dimension N*N')
    plt.ylabel('time per element in nano-seconds')
    plt.legend()
    plt.xlim(0,2100)
    plt.tight_layout()
    plt.savefig(f'{cpuname}_matrix_time.svg')

    fig = plt.figure(figsize=(11, 8))

    plt.plot(dat.N, 2*dat.N**3/(dat.time/1e9), 'x',ms=6,alpha=0.6, label=cpuname, color='tab:red' if 'Ryzen' in cpuname else 'tab:blue')
    plt.xlabel('Matrix<float> size NxN')
    plt.ylabel('Flops/s')
    plt.axvline(caches[0], label ='L1 cache', color='yellow')
    plt.axvline(caches[1], label='L2 cache', color='orange')
    plt.axvline(caches[2], label='half L3 cache',color='tab:blue' if 'Ryzen' in cpuname else 'tab:red')
    plt.legend()
    plt.xlim(0,2100)
    plt.tight_layout()    
    plt.savefig(f'{cpuname}_matrix_flops.svg')


if __name__=='__main__':
    caches = [np.sqrt(384*1024/4),np.sqrt(3*1024*1024/4),np.sqrt(16*1024*1024/4)]
    plot_matrix('../results/results_ryzen.csv','Ryzen 3600',caches)
    caches_intel = [np.sqrt(32*1024/4),np.sqrt(256*1024/4),np.sqrt(5*1024*1024/4)]
    plot_matrix('../results/matrix_creek.csv','Xeon E5-1620',caches_intel)
    plt.show()

