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

MeshPath='Mesh/validation/validation.xml'

datapath='validation/' #folder for results
flow_mode='incompressible' #currently only incompressible is implemented.
"""
n=100
Ss=np.linspace(0,1.8,n)
w0s=np.empty(n)
for i in range(n):
    yo=yaj(MeshPath,flow_mode,datapath,True,0,200,Ss[i],1)

    #Newton solver
    yo.Newton()
    w0s[i]=yo.Getw0()

plt.plot(Ss,w0s)
plt.savefig(datapath+"validation_graph_w0.png")

f_S=inter(Ss,w0s,'quadratic')
print(root(f_S,.89).x)
"""
yo=yaj(MeshPath,flow_mode,datapath,True,-1,200,1,1)
yo.Newton()

#efficiency
yo.BoundaryConditionsPerturbations()
yo.ComputeIndices()
yo.ComputeAM()

#modal analysis
flag_mode=2 #0: save matrix as .mat with file name "savematt"; 1: load result matrix from .mat with file name "loadmatt"; 2: calculate eigenvalues in python  
savematt="Matlab/validation.mat"
loadmatt="Matlab/validation.mat"
yo.Eigenvalues(.05+1j,100,flag_mode,savematt,loadmatt) #shift value; nb of eigenmode
vals=np.loadtxt(yo.dnspath+yo.eig_path+"evals.dat")

#plt.close()
plt.scatter(np.imag(vals),np.real(vals))
plt.savefig(datapath+"validation_eigenvalues.png")