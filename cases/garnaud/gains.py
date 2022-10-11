# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP
from matplotlib import pyplot as plt
from mpi4py.MPI import COMM_WORLD as comm

# Gains
n=50
gains=np.empty(n)
#Sts=np.linspace(.1,1,n)
Sts=[.3]

spyp=SPYP(params,datapath,lambda _: 1e3,1000,nutf,direction_map,0,0,forcingIndicator)
spyp.npyToDatAll({"u":True})
spyp.Q.x.array[spyp.TH0_to_TH]=spyp.U.x.array
boundaryConditionsPerturbations(spyp,0)
# For efficiency, matrices assembled only once
spyp.assembleJNMatrices()
spyp.assembleMRMatrices()
spyp.resolvent(1,Sts)

if comm.rank==0:
    for i,St in enumerate(Sts):
        try:
            gains[i]=np.max(np.loadtxt(f"resolvent/gains_Re=1000_S=0.000_m=0_St={St:00.3f}.txt"))
        except FileNotFoundError:
            print(f"Couldn't load gains for St={St:00.3f}")

    plt.plot(Sts, gains)
    plt.xlabel(r'$St$')
    plt.ylabel(r'$G^{bf}_{opt}(St)$')
    plt.yscale('log')
    plt.savefig("fig6.png")
    plt.close()
    plt.plot(Sts, gains*Sts)
    plt.xlabel(r'$St$')
    plt.ylabel(r'$St G^{bf}_{opt}(St)$')
    plt.yscale('log')
    plt.savefig("fig6_st.png")