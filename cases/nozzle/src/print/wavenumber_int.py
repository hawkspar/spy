# coding: utf-8
"""
Created on Wed Oct  13 17:07:00 2021

@author: hawkspar
"""
from sys import path

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spyp import SPYP
from helpers import j

# Shortcuts
spyp=SPYP(params,data_path,"perturbations",direction_map)
Ss=np.linspace(0,1,6)
dat={'Re':Re,'m':-2,'St':7.3057e-03/2}
source="response"
angles_dir=spyp.resolvent_path+source+"/angles/"

if p0: aes,res=[],[]

for S in Ss:
	dat['S']=S
	save_str=f"_S={S}"
	# Approximate k as division of u
	us=spyp.readMode(source,dat)
	u,_,_=ufl.split(us)
	FS_v = dfx.fem.FunctionSpace(spyp.mesh,ufl.VectorElement("CG",spyp.mesh.ufl_cell(),2))
	Lambda = Function(FS_v)
	Lambda.interpolate(dfx.fem.Expression(ufl.as_vector([ufl.real(u.dx(0)/u/j(spyp.mesh)),
														 dat['m']]),FS_v.element.interpolation_points()))
	spyp.printStuff(angles_dir,"Lambda"+save_str,Lambda)

	# Maximum centrifugal strength
	spyb.loadBaseflow(Re,S,False)
	spyp.interpolateBaseflow(spyb)

	U,_=ufl.split(spyp.Q)
	Sigma = Function(FS_v)
	Sigma.interpolate(dfx.fem.Expression(ufl.as_vector([U[0].dx(1),
												       (U[2]/spyp.r).dx(1)]),FS_v.element.interpolation_points()))
	Sigma.x.array[:]=np.nan_to_num(Sigma.x.array[:])
	spyp.printStuff(angles_dir,"S"+save_str,Sigma)

	# Integration area (shear layer triangle)
	x0,x1,x2=(2,1),(50,12),(50,1)
	def line(x,x1,x2): return x[1]<(x2[1]-x1[1])*(x[0]-x1[0])/(x2[0]-x1[0])+x1[1]
	def kernel(x):
		res = np.zeros_like(x[0])
		res[line(x,x0,x1)]=1
		res[line(x,x0,x2)]=0
		return res
	FS_e = dfx.fem.FunctionSpace(spyp.mesh,ufl.FiniteElement("CG",spyp.mesh.ufl_cell(),2))
	k = Function(FS_e)
	k.interpolate(kernel)
	A=dfx.fem.assemble_scalar(dfx.fem.form(k*ufl.dx)).real # Total weighting surface for averaging

	def dot(u,v=None):
		if v is None: return ufl.sqrt(ufl.real(ufl.dot(u,u))) # compute ||u||
		return ufl.sqrt(ufl.real(ufl.dot(u,v))**2) # compute |<u,v>|
	# Compute dot proSct and orientation
	ae,re = Function(FS_e),Function(FS_e)
	ae.interpolate(dfx.fem.Expression(dot(Lambda,Sigma)*k,FS_e.element.interpolation_points()))
	ae.x.array[:]=np.positive(np.nan_to_num(ae.x.array[:]))
	spyp.printStuff(angles_dir,"pdt"+save_str,ae)
	re.interpolate(dfx.fem.Expression(ae/dot(Lambda)/dot(Sigma),FS_e.element.interpolation_points()))
	re.x.array[:]=np.nan_to_num(re.x.array[:])
	spyp.printStuff(angles_dir,"alignement"+save_str,re)

	# Compute and communicate integrals
	ae,re=dfx.fem.assemble_scalar(dfx.fem.form(ae*ufl.dx)).real,dfx.fem.assemble_scalar(dfx.fem.form(re*ufl.dx)).real
	ae,re,A=comm.gather(ae),comm.gather(re),comm.gather(A)
	if p0:
		A = sum(A)
		ae,re = sum(ae)/A, sum(re)/A
		aes.append(ae)
		res.append(re)
		print("S=",S,"a=",ae,"re=",re,flush=True)

if p0:
	from matplotlib import pyplot as plt
	fig, ax = plt.subplots()
	ax.plot(Ss,res)
	ax.set_xlabel(r'$S$')
	ax.set_ylabel(r'$\frac{1}{A}\int\frac{|\Lambda\cdot\Sigma|}{||\Lambda||||\Sigma||}dA$')
	fig.subplots_adjust(left=0.15)
	fig.savefig(angles_dir+"res.png")

	plt.plot(Ss,aes)
	plt.xlabel(r'$S$')
	plt.ylabel(r'$ae$')
	plt.savefig(angles_dir+"aes.png")