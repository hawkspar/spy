# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from validation_yaj import yaj
import numpy as np
from matplotlib import pyplot as plt
from pdb import set_trace

MeshPath='Mesh/validation/validation.xdmf'
datapath='validation/' #folder for results

# Eigenvalues
yo=yaj(MeshPath,datapath,-1,200,1,1)
# For efficiency, matrix is assembled only once
yo.ComputeAM()

#yo.Resolvent(1,[4.*np.pi])

# Modal analysis
vals_real,vals_imag=np.empty(0),np.empty(0)
# Grid search
for im in np.linspace(-3,3,5):
    for re in np.linspace(-.15,.15,5):
        sigma=re+1j*im
        yo.Eigenvalues(sigma,3) #shift value, nb of eigenmode
        try:
            sig_vals_real,sig_vals_imag=np.loadtxt(datapath+yo.eig_path+"evals"+yo.save_string+".dat",unpack=True)
            vals_real=np.hstack((vals_real,sig_vals_real))
            vals_imag=np.hstack((vals_imag,sig_vals_imag))
        except FileNotFoundError: pass
"""
yo.Eigenvalues(.05+1j,5) #shift value, nb of eigenmode
vals_real,vals_imag=np.loadtxt(datapath+yo.eig_path+"evals"+yo.save_string+".dat",unpack=True)
"""
# Plot them all!
plt.scatter(vals_imag,vals_real,edgecolors='k',facecolors='none')
plt.plot([-1e1,1e1],[0,0],'k--')
#plt.axis([-3,3,-.15,.15])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\sigma$')
plt.savefig(datapath+"eigenvalues"+yo.save_string+".svg")
