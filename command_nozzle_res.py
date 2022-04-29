# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import os
import numpy as np
from spyp import spyp
from mpi4py.MPI import COMM_WORLD

p0=COMM_WORLD.rank==0

MeshPath='../cases/nozzle/nozzle_fine.xdmf'
datapath='nozzle/' #folder for results

# Eigenvalues
spypi=spyp(datapath,1e4,0,0,0,MeshPath)
# For efficiency, matrix is assembled only once
spypi.AssembleMatrices()
# Modal analysis
vals_real,vals_imag=np.empty(0),np.empty(0)
spypi.Resolvent(1,[.2])