import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dat_nullkernel = pd.read_csv('result/nullkernel_timings_out.txt', sep=';', comment=None)
print(dat_nullkernel)
dat_nullkernel.columns = [i.replace('#','').strip() for i in dat_nullkernel.columns]

def plot_startup(df, is_async):

    df = df[df['async'] == is_async]
    plt.figure()
    for block in df.block_count.unique()[:5]:
        _df = df[df.block_count == block]
        plt.plot(_df.thread_count, _df['time/usec'], label=f'blocks: {block}')
    plt.xlabel('thread count')
    plt.ylabel('startup time in microseconds')
    plt.title(f'startup time async={is_async}')
    plt.legend()

    plt.figure()
    for block in df.block_count.unique()[5:]:
        _df = df[df.block_count == block]
        plt.plot(_df.thread_count, _df['time/usec'], label=f'blocks: {block}')
    plt.xlabel('thread count')
    plt.ylabel('startup time in microseconds')
    plt.title(f'startup time async={is_async}')
    plt.legend()


plot_startup(dat_nullkernel, False)
plot_startup(dat_nullkernel, True)
