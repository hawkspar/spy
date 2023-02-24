# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP # Must be after setup
from spy import findStuff
from os.path import isdir
from mpi4py.MPI import COMM_WORLD
from matplotlib import pyplot as plt

# Shorthands
p0=COMM_WORLD.rank==0
load=False
m=2
save_string=f"_Re={Re:d}_S={S}_m={m:d}".replace('.',',')

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
# Modal analysis
if load and p0:
    eigs=np.loadtxt(spyp.eig_path+"evals"+save_string+".dat")
else:
    # Grid search
    X = np.linspace(-2, 2, 20)
    re, im = np.meshgrid(X, X)
    eigs=spyp.eigenvalues(np.flip((re+1j*im).flatten()),1,Re,S,m) # Actual computation shift value, nb of eigenmode
    eigs=np.array(eigs)
if p0:
    # Sum them all, regroup them
    np.savetxt(spyp.eig_path+"evals"+save_string+".dat",eigs)
    # Plot them all!
    fig = plt.figure()
    msk=eigs.real<0
    plt.scatter(eigs.imag[msk], eigs.real[msk], edgecolors='k',facecolors='none')  # Stable eigenvalues
    if eigs[~msk].size>0:
        plt.scatter(eigs.imag[~msk],eigs.real[~msk],edgecolors='k',facecolors='k') # Unstable eigenvalues
    plt.plot([-10,10],[0,0],'k--')
    plt.axis([-2,2,-2,2])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\sigma$')
    plt.savefig("eigenvalues"+save_string+".png")