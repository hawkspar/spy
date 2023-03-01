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
m=2

# Load baseflow
spy = SPY(params, datapath, "baseflow",      direction_map)
spy.loadBaseflow(Re,S)
# Eigenvalues
spyp=SPYP(params, datapath, "perturbations", direction_map)
spyp.Re=Re
# Interpolate and cut baseflow
spyp.interpolateBaseflow(spy)
# BCs
boundaryConditionsPerturbations(spyp,m)
# For efficiency, matrix is assembled only once
spyp.assembleJNMatrices(m)
# Grid search
RE, IM = np.linspace(-5, 5, 41), np.linspace(-.1, .1, 6)
re, im = np.meshgrid(RE, IM)
eigs=set()
for sig in np.flip((re+1j*im).T.flatten()):
    if sig.real>.1: continue #TBM
    # Memoisation
    eig_path=spyp.eig_path+f"values/Re={Re:d}_S={S}_m={m:d}_sig={sig:.2f}".replace('.',',')+".txt"
    if isfile(eig_path):
        if p0: print("Found existing file at "+eig_path+", moving on...",flush=True)
    else: spyp.eigenvalues(sig,10,Re,S,m) # Actual computation shift value, nb of eigenmode