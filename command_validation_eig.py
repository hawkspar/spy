# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import os
import numpy as np
from validation_yaj import yaj
from matplotlib import pyplot as plt

MeshPath='Mesh/validation/validation.xdmf'
datapath='validation/' #folder for results

# Eigenvalues
yo=yaj(MeshPath,datapath,-1,200,1,1)
# For efficiency, matrix is assembled only once
yo.AssembleMatrices()
# Modal analysis
vals_real,vals_imag=np.empty(0),np.empty(0)
# Grid search
for re in np.linspace(.05,.05,1):
    for im in np.linspace(1.2,1.2,1):
        # Memoisation protocol
        sigma=re+1j*im
        closest_file_name=datapath+yo.eig_path+"evals"+yo.save_string+"_sigma="+f"{re:00.3f}"+f"{im:+00.3f}"+"j.dat"
        file_names = [f for f in os.listdir(datapath+yo.eig_path) if f[-3:]=="dat"]
        for file_name in file_names:
            try:
                sigmad = complex(file_name[-12:-4]) # Take advantage of file format 
                fd = abs(sigma-sigmad)#+abs(Re-Red)
                if fd<1e-3:
                    closest_file_name=datapath+yo.eig_path+"evals"+yo.save_string+"_sigma="+f"{np.real(sigmad):00.3f}"+f"{np.imag(sigmad):+00.3f}"+"j.dat"
                    break
            except ValueError: pass
        else:
            yo.Eigenvalues(sigma,5) #Actual computation shift value, nb of eigenmode
        try:
            sig_vals_real,sig_vals_imag=np.loadtxt(closest_file_name,unpack=True)
            vals_real=np.hstack((vals_real,sig_vals_real))
            vals_imag=np.hstack((vals_imag,sig_vals_imag))
        except OSError: pass

# Sum them all, regroup them
np.savetxt(yo.datapath+yo.eig_path+"evals"+yo.save_string+".dat",np.column_stack([vals_real, vals_imag]))
#vals_real, vals_imag=np.loadtxt(yo.datapath+yo.eig_path+"evals"+yo.save_string+".dat",unpack=True)
vals=np.unique((vals_real+1j*vals_imag).round(decimals=3))

# Plot them all!
fig = plt.figure()
ax = fig.add_subplot(111)
msk=vals.real<0
plt.scatter(vals.imag[msk], vals.real[msk], edgecolors='k',facecolors='none') # Stable eigenvalues
if vals[~msk].size>0:
    plt.scatter(vals.imag[~msk],vals.real[~msk],edgecolors='k',facecolors='k')    # Unstable eigenvalues
plt.plot([-1e1,1e1],[0,0],'k--')
plt.axis([-2.5,2.5,-.12,.08])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\sigma$')
plt.savefig(datapath+"eigenvalues"+yo.save_string+".svg")