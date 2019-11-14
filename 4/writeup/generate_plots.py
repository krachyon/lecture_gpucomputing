import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def plot_matrix():
    plt.figure()
    dat = pd.read_csv('../results/results_ryzen.csv',comment='#', header=None)
    dat.columns = ['iters', 'N', 'time']

    dat = dat.sort_values(by=['N']) 
    plt.plot(dat.N, dat.time/(dat.size**2),'x', label='AMD Ryzen 3600')
    
    plt.axvline(np.sqrt(16*1024*1024/4), label='half L3 Cache',color='orange')
    plt.axvline(np.sqrt(3*1024*1024/4), label='L2 Cache', color='red')

    plt.xlabel('matrix<float> dimension N*N')
    plt.ylabel('time per element in nano-seconds')
    plt.legend()
    plt.tight_layout()
    plt.savefig('matrix_time.svg')

    plt.figure()

    plt.plot(dat.N, 2*dat.N**3/(dat.time/1e9), 'x', label='AMD Ryzen 3600')
    plt.xlabel('Matrix<float> size NxN')
    plt.ylabel('Flops/s')
    plt.axvline(np.sqrt(384*1024/4), label ='L1 cache', color='yellow')
    plt.axvline(np.sqrt(16*1024*1024/4), label='half L3 cache',color='orange')
    plt.axvline(np.sqrt(3*1024*1024/4), label='L2 cache', color='red')
    plt.legend()
    plt.tight_layout()
    plt.savefig('matrix_flops.svg')


if __name__=='__main__':
    plot_matrix()
    plt.show()

