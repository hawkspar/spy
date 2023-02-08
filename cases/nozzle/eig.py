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

p0=COMM_WORLD.rank==0
load=False
m=2
save_string=f"_Re={Re:d}_S={S:00.1f}_m={m:d}".replace('.',',')

# Load baseflow
spy = SPY(params,datapath,'baseflow',direction_map)
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
vals_real,vals_imag=np.empty(0),np.empty(0)
if load and p0:
    vals=np.loadtxt(spyp.eig_path+"evals"+save_string+".dat")
    vals_real,vals_imag = vals[:,0],vals[:,1]
else:
    # Grid search
    for re in np.linspace(-2,2,20):
        for im in np.linspace(-2,2,20):
            # Memoisation protocol
            sigma=re+1j*im
            spyp.eigenvalues(sigma,1,Re,S,m) # Actual computation shift value, nb of eigenmode
            if isdir(spyp.eig_path):
                closest_file_name=findStuff(spyp.eig_path,["sig"],[sigma],lambda f: f[-4:]==".txt",False)
                try:
                    if p0:
                        sig_vals_real,sig_vals_imag=np.loadtxt(closest_file_name,unpack=True)
                        vals_real=np.hstack((vals_real,sig_vals_real))
                        vals_imag=np.hstack((vals_imag,sig_vals_imag))
                except OSError: pass # File not found = no eigenvalues
if p0:
    # Sum them all, regroup them
    np.savetxt(spyp.eig_path+"evals"+save_string+".dat",np.column_stack([vals_real, vals_imag]))
    vals=np.unique((vals_real+1j*vals_imag).round(decimals=3))

    # Plot them all!
    fig = plt.figure()
    msk=vals.real<0
    plt.scatter(vals.imag[msk], vals.real[msk], edgecolors='k',facecolors='none')  # Stable eigenvalues
    if vals[~msk].size>0:
        plt.scatter(vals.imag[~msk],vals.real[~msk],edgecolors='k',facecolors='k') # Unstable eigenvalues
    plt.plot([-1e1,1e1],[0,0],'k--')
    plt.axis([-2,2,-2,2])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\sigma$')
    plt.savefig("eigenvalues"+save_string+".png")