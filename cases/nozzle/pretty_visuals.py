# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from spyp import SPYP # Must be after setup

spyp=SPYP(params,datapath,direction_map)
spyp.visualiseRolls(1000,400000,0,1,.1,.5)