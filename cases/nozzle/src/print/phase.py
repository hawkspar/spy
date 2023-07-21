# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from sys import path

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spyp import SPYP
from helpers import dirCreator, grd

spyp= SPYP(params,data_path,"perturbations",direction_map)

dats=[
	  #{"Re":200000,"S":1,"m":-2,"St":7.3057e-03}#,
	  {"Re":200000,"S":1,"m":2,"St":7.3057e-03}#,
	  #{"Re":200000,"S":0,"m":2,"St":0}
]

f = Function(spyp.TH1)
dir=spyp.resolvent_path+"response/phase/"
dirCreator(dir)
for dat in dats:
	us=spyp.readMode("response",dat).split() # Only take axial velocity
	for u,d in zip(us,direction_map.keys()):
		u.x.array[:]=np.arctan2(u.x.array.imag,u.x.array.real)
		u.x.scatter_forward()
		spyp.printStuff(dir,f"Re={dat['Re']}_S={dat['S']}_m={dat['m']}_St={dat['St']}_{d}",u)
		expr=dfx.fem.Expression(u.dx(0),spyp.TH1.element.interpolation_points())
		f.interpolate(expr)
		spyp.printStuff(dir,f"Re={dat['Re']}_S={dat['S']}_m={dat['m']}_St={dat['St']}_{d}_grd_x",f)
		expr=dfx.fem.Expression(u.dx(1),spyp.TH1.element.interpolation_points())
		f.interpolate(expr)
		spyp.printStuff(dir,f"Re={dat['Re']}_S={dat['S']}_m={dat['m']}_St={dat['St']}_{d}_grd_r",f)
		expr=dfx.fem.Expression(dat['m']*dfx.fem.Constant(spyp.mesh,1j)*u/spyp.r,spyp.TH1.element.interpolation_points())
		f.interpolate(expr)
		spyp.printStuff(dir,f"Re={dat['Re']}_S={dat['S']}_m={dat['m']}_St={dat['St']}_{d}_grd_t",f)