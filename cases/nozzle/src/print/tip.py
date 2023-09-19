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

n=250

#X=np.linspace(0,1,n)
#R=np.linspace(1,1.2,n)

"""sets=[({"S":1,"m":-2,"St":.008},np.linspace(.99,1.01,n),np.linspace(.99,1.015,n),[0,.2,.4,.6,.8,1]),
	  #{"S":1,"m": 2,"St":7.3057e-03/2},
	  #{"S":0,"m":-2,"St":7.3057e-03/2}
	  ({"S":0,"m": 2,"St":0},  np.linspace(.99,1.01,n),np.linspace(.99,1.015,n),[0,.2,.4,.6,.8,1])]"""

sets=[#({"S":1,"m":-2,"St":.008},np.linspace(.99,1.01,n),np.linspace(.99,1,n),[0,.2,.4,.6,.8,1]),
	  #{"S":1,"m": 2,"St":7.3057e-03/2},
	  #{"S":0,"m":-2,"St":7.3057e-03/2}
	  ({"S":0,"m": 2,"St":0},  np.linspace(.70,1.1,n),np.linspace(1,1.25,n),[0,.1,.2,.3,.4])]
for set in sets:
	dat,X,R,lvls=set
	dat["Re"]=Re
	# Load baseflow
	spyb.loadBaseflow(Re,dat['S'])
	spyp.interpolateBaseflow(spyb)
	#spyp.save2DQuiver("forcing",dat,1,.95,1.07,300,300,5,150,.9)
	spyp.saveLiftUpTip2("response",dat,X,R,lvls)