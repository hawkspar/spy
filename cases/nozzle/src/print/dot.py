# coding: utf-8
"""
Created on Tue Aug  8 17:27:00 2023

@author: hawkspar
"""
from sys import path
from pickle import dump, load
from itertools import product

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spyp import SPYP
from helpers import j, dirCreator

# Parameter space
Ss =np.linspace(0,1,101)
Sts=np.linspace(0,1,101)
ms=[-2,2]
dats=[{'Re':Re,'S':S,'m':m,'St':St} for S,m,St in product(Ss,ms,Sts)]
source="response"
# Shortcuts
spyp=SPYP(params,data_path,"perturbations",direction_map)
spyp.Re=Re
dir=spyp.resolvent_path+source+"/phase/"
dirCreator(dir)
# Resolvent mask
indic_f = Function(spyp.TH1)
indic_f.interpolate(forcingIndicator)
spyp.assembleNMatrix()
spyp.assembleMRMatrices(indic_f)

# Initialising saving structure
if p0:
	with open(dir+'res.pkl', 'rb') as fp: res=load(fp)
	for S in Ss:
		if not str(S) in res.keys(): res[str(S)]={}
		for m in ms:
			if not str(m) in res[str(S)].keys(): res[str(S)][str(m)]={}
S_save,m_save=-np.inf,-np.inf

FS_v = dfx.fem.FunctionSpace(spyp.mesh,ufl.VectorElement("CG",spyp.mesh.ufl_cell(),2))
Sigma,Lambda = Function(FS_v),Function(FS_v)
k = Function(spyp.TH1)

# Handler
def dot(u,v=None):
	if v is None: return ufl.sqrt(ufl.real(ufl.dot(u,u))) # compute ||u||
	return ufl.sqrt(ufl.real(ufl.dot(u,v))**2) # compute |<u,v>|
def interp(v,v1,v2):
	v.interpolate(dfx.fem.Expression(ufl.as_vector([v1,v2]),FS_v.element.interpolation_points()))
	v.x.array[:]=np.nan_to_num(v.x.array[:])

for dat in dats:
	# Memoisation
	cont=False
	if p0:
		try: res[str(dat['S'])][str(dat['m'])][str(dat['St'])]
		except KeyError: cont=True
	cont = comm.scatter([cont for _ in range(comm.size)])
	if cont: continue
	# Compute only if it changed
	if S_save!=dat['S']:
		comm.barrier()
		spyb.loadBaseflow(Re,dat['S'])
		spyp.interpolateBaseflow(spyb)

		U,_=ufl.split(spyp.Q)
		interp(Sigma,U[0].dx(1),(U[2]/spyp.r).dx(1))
		#spyp.printStuff(angles_dir,"S"+save_str,Sigma)

		# Integration area (shear layer defined as 10% of eddy viscosity)
		k.interpolate(spyb.Nu)
		A=k.x.array[:]
		# Compute max eddy viscosity
		AM = comm.gather(np.max(np.real(A)))
		if p0: AM=max(AM)
		AM = comm.scatter([AM for _ in range(comm.size)])
		# Mask everything below 10%
		k.x.array[A>=.1*AM]=1
		k.x.array[A< .1*AM]=0
		i_r0=np.isclose(spyp.mesh.geometry.x[:,1],0,params['atol']) # Cut out axis of symmetry
		k.x.array[i_r0]=0
		k.x.scatter_forward()
		#spyp.printStuff(dir,"k",k)
		# Total weighting surface for averaging
		A=dfx.fem.assemble_scalar(dfx.fem.form(k*ufl.dx)).real
		A=comm.gather(A)
		if p0: A = sum(A)
		
		S_save=dat['S']

	if m_save!=dat['m']:
		comm.barrier()
		boundaryConditionsPerturbations(spyp,dat['m'])
		# For efficiency, matrices assembled once per Sts
		spyp.assembleJMatrix(dat['m'])
		m_save=dat['m']

	# Resolvent analysis
	spyp.resolvent(1,[dat['St']],Re,dat['S'],dat['m'])
	# Approximate k as division of u
	us=spyp.readMode(source,dat)
	u,_,_=ufl.split(us)
	interp(Lambda,ufl.real(u.dx(0)/u/j(spyp.mesh)),dat['m'])
	

	# Compute dot proSct and orientation
	r = Function(spyp.TH1)
	r.interpolate(dfx.fem.Expression(dot(Lambda,Sigma)/dot(Lambda)/dot(Sigma)*k,spyp.TH1.element.interpolation_points()))
	r.x.array[:]=np.nan_to_num(r.x.array[:])
	#spyp.printStuff(dir,"alignement"+save_str,re)

	# Compute and communicate integrals
	r=dfx.fem.assemble_scalar(dfx.fem.form(r*ufl.dx)).real
	r=comm.gather(r)
	if p0: res[str(dat['S'])][str(dat['m'])][str(dat['St'])] = sum(r)/A

# Writing everything into a double contour plot
if p0:
	with open(dir+'res.pkl', 'wb') as fp: dump(res, fp)
	from matplotlib import pyplot as plt
	fig = plt.figure()
	plt.rcParams.update({'font.size': 20})
	fig.set_size_inches(20,10)
	fig.set_dpi(500)
	gs = fig.add_gridspec(1, 2, wspace=0)
	ax1, ax2 = gs.subplots(sharey=True)
	# Loop of death
	resp,resn=np.empty((len(Ss),len(Sts))),np.empty((len(Ss),len(Sts)))
	for i,S in enumerate(Ss):
		for j,St in enumerate(Sts):
			resp[i,j],resn[i,j]=res[str(S)][str(ms[0])][str(St)],res[str(S)][str(ms[1])][str(St)]
	# Plotting contours
	ax1.contourf(2*Sts,Ss,resp)
	ax1.invert_xaxis()
	c=ax2.contourf(2*Sts,Ss,resn)
	# Common colorbar
	fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(c, cax=cbar_ax)
	ax1.set_xlabel(r'$St$')
	ax2.set_xlabel(r'$St$')
	ax1.set_ylabel(r'$S$')
	fig.savefig(dir+"phase_contour.png")