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

tups=[({"Re":200000,"S":1,"m":-2,"St":7.3057e-03/2},(5.8,1),(42.5,6)),
	  ({"Re":200000,"S":1,"m": 2,"St":7.3057e-03/2},(8.5,1),(37,  5)),
	  ({"Re":200000,"S":0,"m":-2,"St":7.3057e-03/2},(5,  1),(40,  2))]

for tup in tups:
	spyp.saveRPlane("response",tups[0],tup[1],tup[2],1000,500,2)