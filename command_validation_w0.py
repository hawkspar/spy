# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from spyb import spyb
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d as inter
from scipy.optimize import root
from mpi4py.MPI import COMM_WORLD

datapath='validation/' #folder for results

# w_0 graph
n=100
Ss=np.linspace(0,1.8,n)
w0s=np.empty(n)
spybi=spyb(datapath,200,"../cases/validation/validation.xdmf")
for i in range(n):
    # Load existing file
    spybi.Baseflow(i>0,True,Ss[i])
    w0s[i]=spybi.MinimumAxial()

if COMM_WORLD.rank==0:
    # Save velocities
    np.save("../cases/"+datapath+"w0s.npy",w0s)
    # Plot stopping point graph
    plt.plot(Ss,w0s)
    plt.savefig("../cases/"+datapath+"graph_w0.png")
    # Check critical w_0
    f_S=inter(Ss,w0s,'quadratic')
    print(root(f_S,.89).x)
