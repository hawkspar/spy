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
from pdb import set_trace

MeshPath='Mesh/validation/validation.xdmf'
datapath='validation/' #folder for results

# w_0 graph
n=100
Ss=np.linspace(0,1.8,n)
w0s=np.empty(n)
for i in range(n):
    yo=yaj(MeshPath,datapath,0,200,Ss[i],1)

    #Newton solver
    yo.Newton(True) # RUN IN REAL MODE ONLY !
    w0s[i]=np.real(yo.Getw0())

# Save velocities
np.save(datapath+"w0s.npy",w0s)
# Plot stopping point graph
plt.plot(Ss,w0s)
plt.savefig(datapath+"graph_w0.svg")
plt.close()
# Check critical w_0
f_S=inter(Ss,w0s,'quadratic')
print(root(f_S,.89).x)
