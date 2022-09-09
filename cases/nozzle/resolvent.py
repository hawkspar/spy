# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import numpy as np
from setup import *
from spyp import SPYP # Must be after setup

#ms=range(3)
ms=[0]
#Sts=np.linspace(0,1,11)
Sts=[.3]

for m in ms:
	spyp=SPYP(params,datapath,Ref,Re,nutf,direction_map,S,m)#,forcingIndicator)
	boundaryConditionsPerturbations(spyp,m)
	spyp.loadBaseflow(S,Re)
	spyp.sanityCheck()
	spyp.computeSUPG(m)
	# For efficiency, matrices assembled once per Sts
	spyp.assembleJNMatrices()
	spyp.assembleMRMatrices()
	# Resolvent analysis
	spyp.resolvent(1,Sts)