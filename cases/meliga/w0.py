# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from re import search
from os import listdir
from os.path import isfile
from helpers import dirCreator
from scipy.optimize import root
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d as inter

n=19
Ss=np.linspace(0,1.8,n)
N=1000
target_xy = np.zeros((N,2))
target_xy[:,0] = np.linspace(0,5,N)
U,_=spyb.Q.split()
Ux=U.split()[0]

w0s=np.empty(n)
# Baseflow calculation (memoisation)
class_th,u_inlet_th,boundaries=boundaryConditionsBaseflow(spyb,0)
for i,S in enumerate(Ss):
    if isfile(spyb.print_path+f"u_Re={Re:d}_S={S:.2f}".replace('.',',')+".xdmf"): spyb.loadBaseflow(Re,S,False)
    else:
        class_th.S=S
        u_inlet_th.interpolate(class_th)
        spyb.baseflow(Re,S,boundaries)
    w0=np.min(spyb.eval(Ux,target_xy))
    dirCreator(spyb.baseflow_path+"/ws/")
    if p0:
        np.savetxt(spyb.baseflow_path+f"ws/w0_Re={Re:d}_S={S:.2f}".replace('.',',')+".dat",[w0])

# Read them all
if p0:
    dat={}
    file_names = [f for f in listdir(spyb.baseflow_path+"/ws/")]
    for file_name in file_names:
        match = search(r'S=(\d*\,?\d*)',file_name)
        S=match.group(1).replace(',','.')
        if not S in dat.keys(): dat[S]=np.loadtxt(spyb.baseflow_path+"/ws/"+file_name)
    Ss=[float(S) for S in dat.keys()]
    Ss.sort()
    w0s=[dat[f'{S:.2f}']  for S in Ss]
    # Plot stopping point graph
    plt.plot(Ss,w0s)
    plt.xlabel(r'$\max\:u_\theta/\max\:u_x$')
    plt.ylabel(r'$\min\:u_x$')
    plt.savefig("graph_w0.png")
    # Check critical w_0
    f_S=inter(Ss,w0s,'quadratic')
    print(root(f_S,.89).x)