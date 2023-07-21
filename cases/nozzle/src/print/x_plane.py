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

n=1000
m=100

tups=[
	  #({"Re":200000,"S":1,"m":-2,"St":7.3057e-03},.5,.95,1),
	  ({"Re":200000,"S":0,"m":2,"St":0},.75,1,1.2)
]
for tup in tups:
	#spyp.saveXPlane("forcing", tup[0],tup[1],tup[2],1.15,n,n)
	#spyp.saveXPlane("response",tup[0],tup[1],tup[2],1.15,n,n)
	#spyp.saveQuiver("forcing", tup[0],tup[1],tup[2],tup[3],n,n,m,m)
	spyp.saveQuiver("response",tup[0],tup[1],tup[2],tup[3],n,n,30,15)