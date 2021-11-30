# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from validation_yaj import yaj
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace
import os

MeshPath='Mesh/validation/validation.xdmf'
datapath='validation/' #folder for results

# Eigenvalues
yo=yaj(MeshPath,datapath,0,200,1,1)
# For efficiency, matrix is assembled only once
yo.ComputeAM()

#yo.Resolvent(1,[4.*np.pi])
"""
# Modal analysis
vals_real,vals_imag=np.empty(0),np.empty(0)
# Grid search
for re in np.linspace(-.1,.1,5):
    for im in np.linspace(-2,2,5):
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
        else: yo.Eigenvalues(sigma,10) #Actual computation shift value, nb of eigenmode
        try:
            sig_vals_real,sig_vals_imag=np.loadtxt(closest_file_name,unpack=True)
            vals_real=np.hstack((vals_real,sig_vals_real))
            vals_imag=np.hstack((vals_imag,sig_vals_imag))
        except OSError: pass
# Sum them all, regroup them
vals=np.unique((vals_real+1j*vals_imag).round(decimals=3))
np.savetxt(yo.datapath+yo.eig_path+"evals"+yo.save_string+".dat",np.column_stack([vals.real, vals.imag]))
"""
sigma=.05+1j
yo.Eigenvalues(sigma,10) #shift value, nb of eigenmode
vals_real,vals_imag=np.loadtxt(datapath+yo.eig_path+"evals"+yo.save_string+"_sigma="+f"{np.real(sigma):00.3f}"+f"{np.imag(sigma):+00.3f}"+"j.dat",unpack=True)
# Plot them all!
fig = plt.figure()
ax = fig.add_subplot(111)
plt.scatter(vals_imag,vals_real,edgecolors='k',facecolors='none')
#plt.plot([-1e1,1e1],[0,0],'k--')
ax.set_aspect(1)
#plt.axis([-3,3,-.15,.15])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\sigma$')
plt.savefig(datapath+"eigenvalues"+yo.save_string+".svg")