# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from validation_setup import *
from scipy.optimize import root
from mpi4py.MPI import COMM_WORLD
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d as inter
from spyb import SPYB # Must be after setup

# w_0 graph
n=100
Ss=np.linspace(0,1.8,n)
w0s=np.empty(n)
spyb=SPYB(params,datapath,Ref,nutf,boundaryConditionsBaseflow)
for i in range(n):
    # Load existing file
    spyb.baseflow(i>0,True,Ss[i])
    w0s[i]=spyb.minimumAxial()

if COMM_WORLD.rank==0:
    # Plot stopping point graph
    plt.plot(Ss,w0s)
    plt.savefig("../cases/"+datapath+"graph_w0.png")
    # Check critical w_0
    f_S=inter(Ss,w0s,'quadratic')
    print(root(f_S,.89).x)