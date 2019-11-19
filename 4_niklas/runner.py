#! /bin/env python3

import os

kB = 1000
MB = kB * 1000

ITERATIONS = 1000

mems = [kB]

threads = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 750]
bank_threads = [2,4,8,16,32,64,128]
blocks = [1, 2, 3, 4, 5, 10, 20, 30 ,40 ,50, 60]
sizes = [1, 8, 16, 24, 32, 40, 48]
strides = [i for i in range(1,65)]

def global2shared_1b():
    print('\n\n# global2shared_1b\n\n', flush=True)
    with open('results/g2s_1b.csv', 'w') as f:
	    print('#type,size,gDim,bDim,time,bw in GB/s', file=f)
    for thread in threads:
        for size in sizes:
            os.system(f'./bin/memCpy --global2shared -s {size * kB} -i {ITERATIONS} -g 1 -t {thread} >> results/g2s_1b.csv')

def shared2global_1b():
    print('\n\n# shared2global_1b\n\n', flush=True)
    with open('results/s2g_1b.csv', 'w') as f:
	    print('#type,size,gDim,bDim,time,bw in GB/s', file=f)
    for thread in threads:
        for size in sizes:
            os.system(f'./bin/memCpy --shared2global -s {size* kB} -i {ITERATIONS} -g 1 -t {thread} >> results/s2g_1b.csv')


def global2shared():
    print('\n\n# global2shared\n\n', flush=True)
    with open('results/g2s.csv', 'w') as f:
	    print('#type,size,gDim,bDim,time,bw in GB/s', file=f)
    for thread in threads:
        for block in blocks:
            os.system(f'./bin/memCpy --global2shared -s {10 * kB} -i {ITERATIONS} -g {block} -t {thread} >> results/g2s.csv')

def shared2global():
    print('\n\n# shared2global\n\n', flush=True)
    with open('results/s2g.csv', 'w') as f:
	    print('#type,size,gDim,bDim,time,bw in GB/s', file=f)
    for thread in threads:
        for block in blocks:
            os.system(f'./bin/memCpy --shared2global -s {10 * kB} -i {ITERATIONS} -g {block} -t {thread} >> results/s2g.csv')

def register2shared():
    print('\n\n# register2shared\n\n', flush=True)
    with open('results/r2s.csv', 'w') as f:
	    print('#type,size,gDim,bDim,time,bw in GB/s', file=f)
    for thread in threads:
        for block in blocks:
            for size in sizes:
                os.system(f'./bin/memCpy --register2shared -s {size * kB} -i {ITERATIONS} -g {block} -t {thread} >> results/r2s.csv')

def shared2register():
    print('\n\n# shared2register\n\n', flush=True)
    with open('results/s2r.csv', 'w') as f:
	    print('#type,size,gDim,bDim,time,bw in GB/s', file=f)
    for thread in threads:
        for block in blocks:
            for size in sizes:
                os.system(f'./bin/memCpy --shared2register -s {size * kB} -i {ITERATIONS} -g {block} -t {thread} >> results/s2r.csv')

def shared2register_conf():
    print('\n\n# shared2register_conf\n\n', flush=True)
    with open('results/s2r_c.csv', 'w') as f:
	    print('#type,size,gDim,bDim,stride,modulo,clock', file=f)
    for thread in bank_threads:
        for stride in strides: 
                os.system(f'./bin/memCpy --shared2register_conflict -s {512} -i {ITERATIONS} -g 1 -t {thread} -stride {stride} >> results/s2r_c.csv')




if __name__ == '__main__':
    #global2shared_1b()
    #shared2global_1b()
    #global2shared()
    #shared2global()
    #register2shared()
    #shared2register()
    shared2register_conf()



