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
m=75

X=np.linspace(0,1,n)
R=np.linspace(1,1.2,n)

dats=[
	  #{"S":1,"m":-2,"St":7.3057e-03/2},
	  #{"S":1,"m": 2,"St":7.3057e-03/2},
	  #{"S":0,"m":-2,"St":7.3057e-03/2}
	  {"S":0,"m":2,"St":0}
]
for dat in dats:
	dat["Re"]=Re
	#spyp.save2DQuiver("forcing",dat,1,.95,1.07,300,300,5,150,.9)
	spyp.saveLiftUpTip("forcing",dat,X,R,m)