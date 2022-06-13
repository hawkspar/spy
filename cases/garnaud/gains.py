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
n=50
gains=np.empty(n)
Sts=np.linspace(.1,1,n)

"""spyp=SPYP(params,datapath,lambda _: 1e3,nutf,direction_map,0,0,forcingIndicator)
spyp.npyToDatAll()
boundaryConditionsPerturbations(spyp,0)
# For efficiency, matrices assembled only once
spyp.assembleJNMatrices(1000)
spyp.assembleMRMatrices()
spyp.resolvent(1,Sts)"""
for i,St in enumerate(Sts):
    gains[i]=np.max(np.loadtxt(f"resolvent/gains_S=0.000_m=0.00_St={St:00.3f}.txt"))

if COMM_WORLD.rank==0:
    # Plot stopping point graph
    plt.plot(Sts, gains)
    plt.xlabel(r'$St$')
    plt.ylabel(r'$G^{bf}_{opt}(St)$')
    plt.yscale('log')
    plt.savefig("fig6.png")