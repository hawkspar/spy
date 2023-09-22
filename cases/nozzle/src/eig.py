# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from spyp import SPYP
from itertools import product

# Eigenvalues
spyp=SPYP(params, data_path, pert_mesh, direction_map)
spyp.Re=Re

Ss_ref=[1]
ms_ref=[-2]
shifts=[1e-2*(i-5+j*1j-10j) for i,j in product(range(10),range(20))]

spyp.assembleMMatrix()
for S in Ss_ref:
    spyb.loadBaseflow(Re,S)
    # Interpolate and cut baseflow
    spyp.interpolateBaseflow(spyb)
    for m in ms_ref:
        spyp.dofs,spyp.bcs=np.empty(0),[]
        # BCs
        boundaryConditionsPerturbations(spyp,m)
        # For efficiency, matrix is assembled only once
        spyp.assembleLMatrix(m)
        for shift in shifts:
            # Shift invert to near origin
            spyp.eigenvalues(shift,10,Re,S,m,False) # Built-in memoisation