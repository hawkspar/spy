# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
import sys

sys.path.append('/home/shared/src')
sys.path.append('/home/shared/cases')

from nozzle.setup import *

# Parameters grouped here for consistence
Ss_ref =[0]
ms_ref =range(4)
Sts_ref=np.linspace(.05,2,40)/2

datapath='no_nozzle' #folder for results

# Adapted smaller box
def forcingIndicator(x): return (x[0]< x1[0])*slope(line(x0,x1,x))+\
				  				(x[0]>=x1[0])*slope(line(x1,x2,x))