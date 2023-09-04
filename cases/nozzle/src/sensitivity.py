# coding: utf-8
"""
Created on Tue Aug  8 17:27:00 2023

@author: hawkspar
"""
from sys import path
from dolfinx.fem import Function

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spyp import SPYP
from helpers import dirCreator

spyp=SPYP(params,data_path,pert_mesh,direction_map)

dirCreator(spyp.resolvent_path+"sensitivity/")

dats=[{"S":1,"m":-2,"St":.04},
	  {"S":1,"m": 2,"St":.04},
	  {"S":0,"m": 2,"St":0}]

S=Function(spyp.TH1)
for dat in dats:
	r=spyp.readMode("response",dat)
	f=spyp.readMode("forcing",dat)
	for i in range(3):
		for j in range(3):
			S.interpolate(dfx.fem.Expression(ufl.real(f[i]*ufl.conj(r[j])),spyb.TH1.element.interpolation_points()))
			spyp.printStuff(spyp.resolvent_path+"sensitivity/",f"S_{i}{j}",S)