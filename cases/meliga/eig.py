# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from fileinput import filename
import os
import numpy as np
from setup import *
from mpi4py.MPI import COMM_WORLD
from matplotlib import pyplot as plt
from spyp import SPYP # Must be after setup

p0=COMM_WORLD.rank==0
load=True

# Eigenvalues
spyp=SPYP(params, datapath, Ref, nutf, direction_map, 1, -1)
boundaryConditionsPerturbations(spyp,-1)
# For efficiency, matrix is assembled only once
spyp.assembleJNMatrices(100)
# Modal analysis
vals_real,vals_imag=np.empty(0),np.empty(0)
if load and p0:
    vals=np.loadtxt(spyp.eig_path+"evals"+spyp.save_string+".dat")
    vals_real,vals_imag = vals[:,0],vals[:,1]
else:
    # Grid search
    for re in np.linspace(.05,-.1,10):
        for im in np.linspace(-2,2,10):
            # Memoisation protocol
            sigma=re+1j*im
            print(sigma)
            closest_file_name=spyp.eig_path+"evals"+spyp.save_string+"_sigma="+f"{re:00.3f}"+f"{im:+00.3f}"+"j.dat"
            file_names = [f for f in os.listdir(spyp.eig_path) if f[-3:]=="dat"]
            for file_name in file_names:
                print(file_name)
                try:
                    if file_name[-17]=='=':
                        sigmad = complex(file_name[-16:-4]) # Take advantage of file format
                    else:
                        sigmad = complex(file_name[-17:-4]) # Take advantage of file format
                    fd = abs(sigma-sigmad)#+abs(Re-Red)
                    if fd<1e-3:
                        closest_file_name=spyp.eig_path+"evals"+spyp.save_string+"_sigma="+f"{np.real(sigmad):00.3f}"+f"{np.imag(sigmad):+00.3f}"+"j.dat"
                        break
                except ValueError: pass
            else:
                print('bip')
                spyp.eigenvalues(sigma,5) # Actual computation shift value, nb of eigenmode
            try:
                if p0:
                    sig_vals_real,sig_vals_imag=np.loadtxt(closest_file_name,unpack=True)
                    vals_real=np.hstack((vals_real,sig_vals_real))
                    vals_imag=np.hstack((vals_imag,sig_vals_imag))
            except OSError: pass # File not found = no eigenvalues
if p0:
    # Sum them all, regroup them
    np.savetxt(spyp.eig_path+"evals"+spyp.save_string+".dat",np.column_stack([vals_real, vals_imag]))
    vals=np.unique((vals_real+1j*vals_imag).round(decimals=3))

    # Plot them all!
    fig = plt.figure()
    msk=vals.real<0
    plt.scatter(vals.imag[msk], vals.real[msk], edgecolors='k',facecolors='none')  # Stable eigenvalues
    if vals[~msk].size>0:
        plt.scatter(vals.imag[~msk],vals.real[~msk],edgecolors='k',facecolors='k') # Unstable eigenvalues
    plt.plot([-1e1,1e1],[0,0],'k--')
    plt.axis([-2.5,2.5,-.12,.08])
    plt.xlabel(r'$\omega$')
    plt.ylabel(r'$\sigma$')
    plt.savefig("eigenvalues"+spyp.save_string+".png")