# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from sys import path

path.append('/home/shared/src')
path.append('/home/shared/cases/nozzle')

from src.setup import *

# Parameters grouped here for consistence
Ss_ref =[0]
ms_ref =range(4)
Sts_ref=[.05,.2,.6]
"""Sts_ref=np.hstack((np.linspace(0,  .075,6,endpoint=False),np.linspace(.075,.25,6,endpoint=False),
				   np.linspace(.25,.75,11,endpoint=False),np.linspace(.75, 1,  3)))"""

datapath='no_nozzle' #folder for results

# Adapted smaller box
def forcingIndicator(x): return (x[0]< x1[0])*slope(line(x0,x1,x))+\
				  				(x[0]>=x1[0])*slope(line(x1,x2,x))