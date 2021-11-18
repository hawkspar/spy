# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from validation_yaj import yaj
import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d as inter
from scipy.optimize import root

MeshPath='Mesh/validation/validation.xdmf'
datapath='validation/' #folder for results
"""
# w_0 graph
n=100
Ss=np.linspace(0,1.8,n)
w0s=np.empty(n)
for i in range(n):
    yo=yaj(MeshPath,datapath,0,200,Ss[i],1)

    #Newton solver
    yo.Newton()
    w0s[i]=np.real(yo.Getw0())

np.savetxt(datapath+"w0s.dat",w0s)
plt.plot(Ss,w0s)
plt.savefig(datapath+"validation_graph_w0.png")
plt.close()

# Check critical w_0
f_S=inter(Ss,w0s,'quadratic')
print(root(f_S,.89).x)
"""
# Eigenvalues
yo=yaj(MeshPath,datapath,-1,200,1,1)
#yo.Newton() # Unnecessary if already computed
# Efficiency
yo.ComputeAM()

#yo.Resolvent(1,[4.*np.pi])

#modal analysis
#yo.Eigenvalues(.05+1j,20) #shift value; nb of eigenmode
vals_real,vals_imag=np.loadtxt(yo.datapath+yo.eig_path+"evals_S=1.000_m=-1.dat",unpack=True)
plt.scatter(vals_imag,vals_real,edgecolors='k',facecolors='none')
plt.plot([-1e1,1e1],[0,0],'k--')
plt.axis([-3,3,-.15,.15])
plt.xlabel(r'$\omega$')
plt.ylabel(r'$\sigma$')
plt.savefig(datapath+"validation_eigenvalues.png")