# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from setup import *
from spyp import SPYP # Must be after setup

S,m,St=1,-2,.05
spyp=SPYP(params,data_path,pert_mesh,direction_map)
for dir in range(3):
	k=spyp.readK("response",Re,S,m,St,dir)
	spyp.printStuff("./",f"k_{dir}_Re={Re}_S={S:.1f}_m={m}_St={St:.2f}".replace('.',','),k)