# coding: utf-8
"""
Created on Wed Aug  02 10:00:00 2023

@author: hawkspar
"""
from sys import path

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spyp import SPYP
from scipy.optimize import fmin
from scipy.interpolate import LinearNDInterpolator as lin

# Shortcuts
spyp=SPYP(params,data_path,"perturbations",direction_map)
dir='/home/shared/cases/nozzle/baseflow/characteristics/'

# Load baseflow, extract 2 main components
spyb.loadBaseflow(Re,1,False)

# Handlers
FS = dfx.fem.FunctionSpace(spyb.mesh,ufl.FiniteElement("CG",spyb.mesh.ufl_cell(),2))
O,P,dU = Function(FS),Function(FS),Function(FS)
U,_=spyb.Q.split()
Ux,_,Ut=ufl.split(U)
r=spyb.r

def interp(F,expr): F.interpolate(dfx.fem.Expression(expr,FS.element.interpolation_points()))

interp(O,Ut/r) # Angular velocity
interp(P,-2*O*O.dx(1)*r/((O.dx(1)*r)**2+Ux.dx(1)**2)*(O.dx(1)*(r*Ut).dx(1)+Ux.dx(1)**2))
interp(dU,Ux.dx(1)**2+Ut.dx(1)**2) # Gradient

spyb.printStuff(dir,"dU_norm",dU)
spyb.printStuff(dir,"phi",P)

nx,nr=500,500
Xs=np.linspace(1,50,nx)
Rs=np.linspace(0,1,nr)
Rg=np.outer(Xs*10/50,Rs).flatten()
Xg=np.tile(Xs.reshape(-1,1),(1,nr)).flatten()
XR=np.array([[x,r] for x,r in zip(Xg,Rg)])
dU=spyb.eval(dU,XR)
XRo,P=spyb.eval(P,XR,XR)

dU_M,P_M=np.ones(nx),np.ones(nx)
if p0:
	print("Evaluation over, computing maximums")
	dU_lin,P_lin=lin(XR,dU),lin(XR,P)
	for i,x in enumerate(Xs):
		dU_M[i]=fmin(lambda r: -dU_lin(x,r),dU_M[i-1],disp=0)[0]
		P_M[i] =fmin(lambda r: -P_lin(x,r), P_M[i-1], disp=0)[0]
	print("Drawing critical line")
comm.Bcast(dU_M, root=0)
comm.Bcast(P_M,  root=0)

dU_crt,P_crt=Function(FS),Function(FS)
def crit_line_f(M,x):
	f=np.zeros_like(x[0])
	for i in range(nx-1):
		msk=(Xs[i]<x[0])*(x[0]<Xs[i+1])
		f[msk]=np.isclose(x[1,msk],(M[i+1]-M[i])/(Xs[i+1]-Xs[i])*(x[0,msk]-Xs[i])+M[i],atol=1e-3)
	return f
dU_crt.interpolate(lambda x: crit_line_f(dU_M,x))
P_crt.interpolate( lambda x: crit_line_f(P_M, x))
spyb.printStuff(dir,"crit_line_KH",dU_crt)
spyb.printStuff(dir,"crit_line_cent",P_crt)