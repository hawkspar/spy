# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP # Must be after setup
from spyb import SPYB
from mpi4py.MPI import COMM_WORLD as comm

ms=range(-5,-3)
Sts=np.hstack((np.linspace(1,.2,20,endpoint=False),np.linspace(.2,.1,5,endpoint=False),np.linspace(.1,.01,10)))
Ss=[0,1]
"""ms=[3]
Sts=[.01]
Ss=[1]"""
Res=[1000]
nut=400000
stab=False

for m in ms:
	spyb=SPYB(params,datapath,Ref,nutf,direction_map,InletAzimuthalVelocity, True)
	spyb.loadBaseflow(S,Re,True)
	boundaryConditionsBaseflow(spyb)
	spyb.sanityCheckBCs()
	if comm.rank==0: print("BCs fixed",flush=True)
	spyb.smoothen(.1)
	spyb.baseflow(Re,0,False)
	spyb.sanityCheck()
	"""spyp=SPYP(params,datapath,Ref,Re,nutf,direction_map,S,m)#,forcingIndicator)
	spyp.loadBaseflow(S,Re)
	spyp.smoothen(.1)
	spyp.sanityCheck()
	boundaryConditionsPerturbations(spyp,m)
	#spyp.sanityCheckBCs()
	spyp.computeSUPG(m)
	# For efficiency, matrices assembled once per Sts
	spyp.assembleJNMatrices()
	spyp.assembleMRMatrices()
	# Resolvent analysis
	spyp.resolvent(1,Sts)"""
