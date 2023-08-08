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
m=10

kwargs=[
	  #({"S":1,"m":-2,"St":7.3057e-03/2},.5,  .9,1,.9),
	  #{'dat':{"S":1,"m":-2,"St":7.3057e-03/2},'x':1.0001,'r_min':.99,'r_max':1.01,'step':100,'s':100,'o':.99},
	  #({"S":1,"m": 2,"St":7.3057e-03/2},.5,  .9,1,.9),
	  #({"S":1,"m": 2,"St":7.3057e-03/2},1.0001,.99,1.01,.99),
	  #({"S":0,"m":-2,"St":7.3057e-03/2},.5,  .9,1,.9),
	  #({"S":0,"m":-2,"St":7.3057e-03/2},1.0001,.99,1.01,.99),
	  #({"S":0,"m": 2,"St":7.3057e-03/2},.5,  .9,1,.9),
	  #({"S":0,"m": 2,"St":7.3057e-03/2},1,.99,1.01,.99),
	  #({"S":1,"m":-2,"St":7.3057e-03},  .5,  .9,1,.9),
	  #({"S":1,"m": 2,"St":7.3057e-03},  .5,  .9,1,.9),
	  {'dat':{"S":0,"m":2,"St":0},'x':.4,'r_min':1,'r_max':1.15,'step':25,'s':30,'o':.9}
]
for kwarg in kwargs:
	kwarg['dat']['Re']=Re
	kwarg['n_th'],kwarg['n_r']=n,n
	#spyp.saveXPlane("forcing", tup[0],tup[1],tup[2],tup[3],n,n,tup[4])
	#spyp.saveXPlane("response",tup[0],tup[1],tup[2],tup[3],n,n)
	spyp.save2DQuiver("forcing", **kwarg)
	spyp.save2DQuiver("response",**kwarg)