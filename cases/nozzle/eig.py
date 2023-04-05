# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP # Must be after setup
from os.path import isfile
from mpi4py.MPI import COMM_WORLD

# Shorthands
p0=COMM_WORLD.rank==0
ms=range(-2,3)
Ss=np.linspace(0,1.3,14)

# Load baseflow
spy = SPY(params, datapath, "baseflow",      direction_map)
# Eigenvalues
spyp=SPYP(params, datapath, "perturbations", direction_map)
spyp.Re=Re

for S in Ss:
    spy.loadBaseflow(Re,S)
    # Interpolate and cut baseflow
    spyp.interpolateBaseflow(spy)
    for m in ms:
        spyp.dofs,spyp.bcs=np.empty(0),[]
        # BCs
        boundaryConditionsPerturbations(spyp,m)
        # For efficiency, matrix is assembled only once
        spyp.assembleJNMatrices(m)
        # Shift invert to origin
        sig=1e-6
        eig_path=spyp.eig_path+f"values/Re={Re:d}_S={S}_m={m:d}_sig={sig:.2f}".replace('.',',')+".txt"
        if isfile(eig_path):
            if p0: print("Found existing file at "+eig_path+", moving on...",flush=True)
        else: spyp.eigenvalues(sig,10,Re,S,m) # Actual computation shift value, nb of eigenmode