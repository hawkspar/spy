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

load = True
n=100
Ss=np.linspace(0,1.8,n)
spyb=SPYB(params,datapath,Ref,nutf,direction_map,InletAzimuthalVelocity)

if not load:
    w0s=np.empty(n)
    # Baseflow calculation (memoisation)
    boundaryConditionsBaseflow(spyb)
    for i in range(n):
        # Load existing file
        spyb.baseflow(i>0,True,Ss[i])
        w0s[i]=spyb.minimumAxial()
    np.savetxt(spyb.baseflow_path+"w0s.dat",w0s)
else: w0s=np.loadtxt(spyb.baseflow_path+"w0s.dat")

if COMM_WORLD.rank==0:
    # Plot stopping point graph
    plt.plot(Ss,w0s)
    plt.xlabel(r'$\max\:u_\theta/\max\:u_x$')
    plt.ylabel(r'$\min\:u_x$')
    plt.savefig("graph_w0.png")
    # Check critical w_0
    f_S=inter(Ss,w0s,'quadratic')
    print(root(f_S,.89).x)