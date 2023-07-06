# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from spyp import SPYP

# Eigenvalues
spyp=SPYP(params, data_path, pert_mesh, direction_map)
spyp.Re=Re

spyp.assembleNMatrix()
for S in Ss_ref:
    spyb.loadBaseflow(Re,S)
    # Interpolate and cut baseflow
    spyp.interpolateBaseflow(spyb)
    for m in ms_ref:
        spyp.dofs,spyp.bcs=np.empty(0),[]
        # BCs
        boundaryConditionsPerturbations(spyp,m)
        # For efficiency, matrix is assembled only once
        spyp.assembleJMatrix(m)
        # Shift invert to near origin
        spyp.eigenvalues(1e-4,10,Re,S,m) # Built-in memoisation