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

dat={"Re":200000,"S":1,"m":-2,"St":7.3057e-03/2}
spyp.visualiseRPlane("response",dat,1,40,1,1,1000,1000)