import pandas as pd
import matplotlib.pyplot as plt

dat=pd.read_csv('total.csv',comment='#', header=None)
dat.columns="N_elem,N_block,dtype,method,t_tot".split(',')

std=dat[dat.method=='std']

loglog(std.N_elem, std.t_tot)                                                                                                                                                                                                                                                                                                                     
thrust=dat[dat.method=='thrust']                                                                                                                                                                                                                                                                                                                  
loglog(thrust.N_elem, thrust.t_tot)                                                                                                                                                                                                                                                                                                               

naive=dat[dat.method=='cuda_naive']
shared=dat[dat.method=='cuda_shared']

loglog(naive.N_elem, naive.t_tot,'x')                                                                                                                                                                                                                                                                                                               
loglog(shared.N_elem, shared.t_tot,'x')                                                                                                                                                                                                                                                                                                               
