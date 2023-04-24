# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyb import SPYB # Must be after setup
from spyp import SPYP
from os.path import isfile
from mpi4py.MPI import COMM_WORLD

# Shorthands
p0=COMM_WORLD.rank==0

# Load baseflow
spyb=SPYB(params, datapath, base_mesh, direction_map)
# Eigenvalues
spyp=SPYP(params, datapath, pert_mesh, direction_map)
spyp.Re=Re

for S in Ss_ref:
    spyb.loadBaseflow(Re,S)
    spyb.smoothenNu(1e-4)
    # Interpolate and cut baseflow
    spyp.interpolateBaseflow(spyb)
    for m in ms_ref:
        spyp.dofs,spyp.bcs=np.empty(0),[]
        # BCs
        boundaryConditionsPerturbations(spyp,m)
        # For efficiency, matrix is assembled only once
        spyp.assembleJNMatrices(m)
        # Shift invert to origin
        sig=1e-4
        eig_path=spyp.eig_path+f"values/Re={Re:d}_S={S}_m={m:d}_sig={sig:.2f}".replace('.',',')+".txt"
        if not isfile(eig_path): spyp.eigenvalues(sig,10,Re,S,m) # Actual computation shift value, nb of eigenmode
        elif p0: print("Found existing file at "+eig_path+", moving on...",flush=True)