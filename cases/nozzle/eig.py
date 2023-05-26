# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP

# Eigenvalues
spyp=SPYP(params, data_path, pert_mesh, direction_map)
spyp.Re=Re

for S in Ss_ref:
    spyb.loadBaseflow(Re,S)
    # Interpolate and cut baseflow
    spyp.interpolateBaseflow(spyb)
    for m in ms_ref:
        spyp.dofs,spyp.bcs=np.empty(0),[]
        # BCs
        boundaryConditionsPerturbations(spyp,m)
        # For efficiency, matrix is assembled only once
        spyp.assembleJNMatrices(m)
        # Shift invert to near origin
        sig=1e-4
        eig_path=spyp.eig_path+f"values/Re={Re:d}_S={S}_m={m:d}_sig={sig:.2f}".replace('.',',')+".txt"
        spyp.eigenvalues(sig,10,Re,S,m) # Built-in memoisation