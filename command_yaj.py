"""
Created on Fri Apr  9 16:23:28 2021

@author: cwang
"""
from yaj import yaj
import numpy as np

MeshPath='Mesh/jet/jet.xml'

datapath='jet_incompressible/' #folder for results


import_flag=1 #1: import base flow from file #0: not import
flow_mode='incompressible' #currently only incompressible is implemented.
yo=yaj(MeshPath,flow_mode,datapath,import_flag)

#Newton solver
yo.Newton() 

#resolvent analysis
yo.Resolvent(1,[0.4*np.pi]) #nb of resolvent mode(currently output implemented only for nb=1); frequency: St*pi

#modal analysis
flag_mode=2 #0: save matrix as .mat with file name "savematt"; 1: load result matrix from .mat with file name "loadmatt"; 2: calculate eigenvalues in python  
savematt="Matlab/incompressible_jet.mat"
loadmatt="Matlab/incompressible_jet.mat"
yo.Eigenvalues(-0.2+1j,10,flag_mode,savematt,loadmatt) #shift value; nb of eigenmode

