import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def clean_header(df):
    df.columns = [i.replace('#', '').strip() for i in df.columns]


def plot_startup(is_async):
    dat_nullkernel = pd.read_csv('result/nullkernel_timings_out.txt', sep=';', comment=None)
    print(dat_nullkernel)
    clean_header(dat_nullkernel)
    df = dat_nullkernel

    df = df[df['async'] == is_async]
    plt.figure()
    for block in df.block_count.unique()[:5]:
        _df = df[df.block_count == block]
        plt.plot(_df.thread_count, _df['time/usec'], label=f'blocks: {block}')
    plt.xlabel('thread count')
    plt.ylabel('startup time in microseconds')
    plt.title(f'startup time async={is_async}')
    plt.legend()
    plt.savefig(f'plots/0-5_async_{is_async}.svg')

    plt.figure()
    for block in df.block_count.unique()[5:]:
        _df = df[df.block_count == block]
        plt.plot(_df.thread_count, _df['time/usec'], label=f'blocks: {block}')
    plt.xlabel('thread count')
    plt.ylabel('startup time in microseconds')
    plt.title(f'startup time async={is_async}')
    plt.legend()
    plt.savefig(f'plots/5-10_async_{is_async}.svg')


def plot_wait():
    plt.figure()

    data = pd.read_csv('result/wait_out.txt', sep=';', comment=None)
    clean_header(data)

    ticks = data.waitticks
    t = data['elapsed time in nanoseconds']

    baseline = t[0:100].mean()
    idx = abs(t-2*baseline).argsort()[0]
    print(ticks[idx])

    plt.xlim(0,6000)
    plt.ylim(0,6000)
    plt.axhline(baseline, color='g', label=f'baseline {baseline} ms')
    plt.axhline(baseline*2, color='g', label=f'double {2*baseline} ms')
    plt.axvline(ticks[idx], color='r', label=f'{ticks[idx]} ticks')
    plt.plot(ticks, t)
    plt.xlabel('clock ticks waited')
    plt.ylabel('kernel runtime in nanoseconds')
    plt.legend()
    plt.savefig(f'plots/wait.svg')
    return data


def plot_memory():
    plt.figure()

    data = pd.read_csv('result/memory_nvidia.txt', sep=';', comment=None)
    clean_header(data)
    # PCIE 2.0 x16: 8 GB/s
    # data is bytes per milisecond
    theoretical_maximum = data.Size/(1024*1024*1024*8/1e3)

    plt.loglog(data.Size, data.H2DPage, label='H2DPage')
    plt.loglog(data.Size, data.H2DPin, label='H2DPin')
    plt.loglog(data.Size, data.D2HPage, label='D2HPage')
    plt.loglog(data.Size, data.D2HPin, label='D2HPin')
    plt.loglog(data.Size, theoretical_maximum, label='maximum throughput')
    plt.xlabel('data size in bytes')
    plt.ylabel('time to copy in ms')
    plt.title('modified throughput example from nvidia')

    plt.legend()
    plt.savefig(f'plots/memory.svg')

    data = pd.read_csv('result/memory.txt', sep=';', comment=None)
    clean_header(data)
    # data is bytes per nanosecond
    theoretical_maximum = data.Size/(1024*1024*1024*8/1e9)

    plt.figure()
    plt.loglog(data.Size, data.H2DPage, label='H2DPage')
    plt.loglog(data.Size, data.H2DPin, label='H2DPin')
    plt.loglog(data.Size, data.D2HPage, label='D2HPage')
    plt.loglog(data.Size, data.D2HPin, label='D2HPin')
    plt.loglog(data.Size, theoretical_maximum, label='maximum throughput')
    plt.xlabel('data size in bytes')
    plt.ylabel('time to copy in ns')
    plt.title('first attempt at measuring copy latency')
    plt.legend()
    plt.savefig(f'plots/memory_error.svg')


plot_startup(False)
plot_startup(True)
dat = plot_wait()
plot_memory()
