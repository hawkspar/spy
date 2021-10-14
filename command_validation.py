# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from validation_yaj import yaj
import numpy as np

MeshPath='Mesh/validation/validation.xml'

datapath='validation/' #folder for results


flow_mode='incompressible' #currently only incompressible is implemented.
yo=yaj(MeshPath,flow_mode,datapath,False,0,200,1,10)

#Newton solver
yo.Newton()

#efficiency
yo.BoundaryConditionsPerturbations()
yo.ComputeIndices()
yo.ComputeAM()

#resolvent analysis
yo.Resolvent(1,[.4*np.pi]) #nb of resolvent mode(currently output implemented only for nb=1); frequency: St*pi

#modal analysis
"""
flag_mode=2 #0: save matrix as .mat with file name "savematt"; 1: load result matrix from .mat with file name "loadmatt"; 2: calculate eigenvalues in python  
savematt="Matlab/incompressible_jet.mat"
loadmatt="Matlab/incompressible_jet.mat"
yo.Eigenvalues(-0.2+1j,10,flag_mode,savematt,loadmatt) #shift value; nb of eigenmode
"""