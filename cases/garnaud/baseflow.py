# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from spyb import SPYB # Must be after setup

for Re in [10, 50, 100, 500, 1000]:
	spyb=SPYB(params,datapath,lambda _: Re,nutf,direction_map,InletAzimuthalVelocity)
	# Baseflow calculation
	boundaryConditionsBaseflow(spyb)
	spyb.baseflow(Re,0,False,True,baseflowInit=baseflowInit)