# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from spyb import SPYB # Must be after setup

spyb=SPYB(params,datapath,"garnaud",direction_map)
spyb.Nu=0
# Baseflow calculation
boundaryConditionsBaseflow(spyb)
spyb.Re=10
spyb.baseflow(spyb.Re,S,save=False,baseflowInit=baseflowInit)
for cur_Re in [50, 250, Re]:
	spyb.Re=cur_Re
	spyb.baseflow(cur_Re,S,save=cur_Re==Re)