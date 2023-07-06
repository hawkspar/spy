# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from sys import path

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spyp import SPYP

spyp= SPYP(params,data_path,"perturbations",direction_map)

dats=[{"Re":200000,"S":1,"m":-2,"St":7.3057e-03/2},
	  {"Re":200000,"S":1,"m": 2,"St":7.3057e-03/2},
	  {"Re":200000,"S":0,"m":-2,"St":7.3057e-03/2}]
for dat in dats:
	spyp.visualiseRPlane("response",dat,(7.5,1),(45,6.2),1000,500,5)