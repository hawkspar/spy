# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from os import listdir
from matplotlib import pyplot as plt

from setup import *
from spyp import SPYP # Must be after setup

m, S = -1, 1

# Eigenvalues
spyp=SPYP(params, datapath, "validation", direction_map)
spyp.Re=np.inf
# Interpolate
spyb.loadBaseflow(Re,S,False)
spyp.interpolateBaseflow(spyb)
spyp.Nu.interpolate(lambda x: 1/sponged_Reynolds(x))

# For efficiency, matrix is assembled only once
boundaryConditionsPerturbations(spyp,m)
spyp.assembleJNMatrices(m)

spyp.sanityCheck()

# Grid search
for re in [.05]:#np.linspace(.05,-.1,4).round(decimals=3):
    for im in [1]:#np.linspace(-1,1,4).round(decimals=3):
        #if re>0 and abs(im-1)>.2: continue
        spyp.eigenvalues(re+1j*im,20,Re,S,m) # Actual computation shift value, nb of eigenmodes
if p0:
    # Read them all, regroup them
    vals=[]
    for file in listdir(spyp.eig_path+"values/"):
        if file[-3:]=="txt":
            a=np.loadtxt(spyp.eig_path+"values/"+file,dtype=complex)
            if   a.size==1: vals.append(a)
            elif a.size> 1: vals.extend(list(a))
    vals=np.unique(np.array(vals).round(decimals=3))

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
    plt.savefig("eigenvalues"+f"_Re={Re:d}_m={m:d}_S={S:.2f}".replace(',','.')+".png")