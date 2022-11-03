# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP # Must be after setup

ms=range(3)
Sts=np.linspace(.1,.2,10,endpoint=False)
Ss=[0,.1]
Res=[1000]
nut=400000
stab=False

spyp=SPYP(params,datapath,direction_map,forcingIndicator)
for Re in Res:
	Ref(spyp,Re)
	for S in Ss:
		spyp.loadBaseflow(Re,nut,S) # Don't load pressure
		nutf(spyp,nut,S)
		for m in ms:
			boundaryConditionsPerturbations(spyp,m)
			spyp.stabilise(m)
			# For efficiency, matrices assembled once per Sts
			spyp.assembleJNMatrices(m,stab)
			spyp.assembleMRMatrices(stab)
			# Resolvent analysis
			spyp.resolvent(4,Sts,Re,nut,S,m,True)