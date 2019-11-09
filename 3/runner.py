#! /bin/env python3

import os

kB = 8 * 1024
MB = kB * 1024

mems = [kB]

# this should go to 0.25 GB
for i in range(18):
    mems.append(mems[-1] * 2)

# run Memcopy benchmark
print('\n\nmemcopy, paged\n\n')
for mem in mems:
    os.system(f'./main --memcpy -s {mem} -im 10')

print('\n\nmemcopy, pinned\n\n')
for mem in mems:
    os.system(f'./main --memcpy -s {mem} -p -im 10')

# run coalesced access, arbitrarily choose 16 Megs
threads = [1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 750]
blocks = [1, 2, 3, 4, 5, 10, 20, 32]

print('\n\ncoalesced\n\n')
for thread in threads:
    for block in blocks:
        os.system(f'./main --global-coalesced -s {16 * MB} -i 10 -g {block} -t {thread}')

print('\n\nstride\n\n')
for thread in threads:
    for block in blocks:
        for stride in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            os.system(f'./main --global-stride -s {16 * MB * stride} -i 10 -g {block} -t {thread} --stride{stride}')

print('\n\noffset\n\n')
for thread in threads:
    for block in blocks:
        for offset in [2, 3, 4, 5, 6, 7, 8, 9, 10]:
            os.system(f'./main --global-ofset -s {16 * MB + offset} -i 10 -g {block} -t {thread} --offset{offset}')
