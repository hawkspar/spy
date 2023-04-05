# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP
from matplotlib import pyplot as plt
from dolfinx.fem import FunctionSpace
from mpi4py.MPI import COMM_WORLD as comm

# Gains
n=50
gains=np.empty(n)
Sts=np.linspace(.1,1,n)

spyp=SPYP(params,datapath,"garnaud",direction_map)
spyp.Nu,spyp.Re=0,Re
boundaryConditionsPerturbations(spyp,0)
spyp.loadBaseflow(S,Re,False)
spyp.sanityCheckU()
# For efficiency, matrices assembled only once
spyp.assembleJNMatrices(0)
FE_constant=ufl.FiniteElement("CG",spyp.mesh.ufl_cell(),1)
W = FunctionSpace(spyp.mesh,FE_constant)
indic = Function(W)
indic.interpolate(forcingIndicator)
spyp.assembleMRMatrices(indic)
spyp.resolvent(1,Sts,Re,S,0)

if comm.rank==0:
    for i,St in enumerate(Sts):
        try:
            gains[i]=np.max(np.loadtxt(f"resolvent/gains/Re=1000_S=0_m=0_St={St:.2f}".replace('.',',')+".txt"))
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