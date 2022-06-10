# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP
from mpi4py.MPI import COMM_WORLD
from matplotlib import pyplot as plt

# Gains
n=100
gains=np.empty(n)
Sts=np.linspace(.1,1,n)

spyp=SPYP(params,datapath,Ref,nutf,direction_map,0,0)
# For efficiency, matrices assembled only once
spyp.assembleJNMatrices()
spyp.assembleMRMatrices()
spyp.resolvent(1,Sts)
for i,St in enumerate(Sts):
    gains[i]=np.loadtxt(f"resolvent/gains_S=0.000_m=0.00_St={St:00.3f}.dat")

if COMM_WORLD.rank==0:
    # Plot stopping point graph
    plt.plot(Sts, gains)
    plt.xlabel(r'$St$')
    plt.ylabel(r'$G^{bf}_{opt}$')
    plt.savefig("fig6.png")