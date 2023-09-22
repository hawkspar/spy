# coding: utf-8
"""
Created on Tue Aug  8 17:27:00 2023

@author: hawkspar
"""
from sys import path
from os.path import isfile
from pickle import dump, load
from itertools import product
from scipy.optimize import fmin
from scipy.special import jn_zeros
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, griddata
from matplotlib.colors import SymLogNorm

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spyp import SPYP
from helpers import j, dirCreator, findStuff

# Parameter space
Ss =np.linspace(0,1,11)
Sts=np.unique(np.hstack([np.linspace(0,1,11),np.linspace(0,.02,5)]))
Sts.sort()
ms=[2,-2]
dats=[{'Re':Re,'S':S,'m':m,'St':St} for S,m,St in product(Ss,ms,Sts)]
source="response"
prt=False # Use sparingly, may tank performance
nc=50 # Step for intermediate saves
# Shortcuts
spyp=SPYP(params,data_path,"perturbations",direction_map)
spyp.Re=Re
dir=spyp.resolvent_path+source+"/phase/"
dirCreator(spyp.resolvent_path)
dirCreator(spyp.resolvent_path+source)
dirCreator(dir)
# Resolvent mask
indic_f = Function(spyp.TH1)
indic_f.interpolate(forcingIndicator)
spyp.assembleMMatrix()
spyp.assembleWBRMatrices(indic_f)
# Number of inertial waves
nt=10

res={"avgs":{},"devs":{},"gains":{},"guptas":{}}
#res={"guptas":{}}

# Triggers computations
calc_avgs="avgs" in res.keys() or "devs" in res.keys()
calc_devs="devs" in res.keys()
calc_gains="gains" in res.keys()
calc_guptas="guptas" in res.keys()

# Initialising saving structure
if p0:
	for name in res.keys():
		if isfile(dir+name+".pkl"):
			with open(dir+name+'.pkl', 'rb') as fp: res[name]=load(fp)
		for S in Ss:
			if not S in res[name].keys(): res[name][S]={}
			for m in ms:
				if not m in res[name][S].keys(): res[name][S][m]={}
S_save,m_save=-np.inf,-np.inf

FS_v = dfx.fem.FunctionSpace(spyp.mesh,ufl.VectorElement("CG",spyp.mesh.ufl_cell(),2))
Sigma,Lambda = Function(FS_v),Function(FS_v)
k,kernel,chi,f,gupta,R_rot = Function(spyp.TH1),Function(spyp.TH1),Function(spyp.TH1),Function(spyp.TH1),Function(spyp.TH1),Function(spyp.TH1)
# Handle
r=spyp.r

# Handler
def dot(u,v=None):
	if v is None: return ufl.sqrt(ufl.real(ufl.dot(u,u))) # compute ||u||
	return ufl.sqrt(ufl.real(ufl.dot(u,v))**2) # compute |<u,v>|
def interp_scalar(v,v1,imag=False):
	v.interpolate(dfx.fem.Expression(v1,spyp.TH1.element.interpolation_points()))
	v.x.array[:]=np.nan_to_num(v.x.array[:])
	if imag: v.x.array[:]=v.x.array[:].imag
def interp_vector(v,v1,v2):
	v.interpolate(dfx.fem.Expression(ufl.as_vector([v1,v2]),FS_v.element.interpolation_points()))
	v.x.array[:]=np.nan_to_num(v.x.array[:]).real

#try:
c=0
for dat in dats:
	S,m,St=dat['S'],dat['m'],dat['St']
	save_str="_S="+str(S)+f"_m={m:d}_St={St:.3f}"
	# Memoisation
	cont=True
	if p0:
		try:
			for name in res.keys(): res[name][S][m][St]
			print("Found data in dictionary, moving on...",flush=True)
		except KeyError: cont=False
	cont = comm.scatter([cont for _ in range(comm.size)])
	if cont: continue
	# Load only if it changed from last computation
	if S_save!=S:
		comm.barrier()
		spyb.loadBaseflow(Re,S) # Sometimes unecessary to load nut
		spyp.interpolateBaseflow(spyb)
		U,_=ufl.split(spyp.Q)
		if calc_avgs or calc_devs: interp_vector(Sigma,U[0].dx(1),(U[2]/r).dx(1))
		# Compute rotating radius
		if prt:
			n=1000
			X=np.linspace(0,25,n//2)
			R=np.linspace(0,1,n)
			X_m,R_m=np.meshgrid(X,R)
			XR = np.array([[x,r] for x,r in zip(X_m,R_m)])
			dUt = spyb.eval(U[2].dx(1),XR)
			Rc=np.empty(n)
			if p0:
				dUt.reshape(X.size,-1).real.T
				for i in range(n//2):
					Rc[i]=fmin(interp1d(R,np.abs(dUt[i])),.9-1/20*X[i]*(X[i]>1))
			Rc = comm.scatter([Rc for _ in range(comm.size)])
			def R_f(x):
				R=np.zeros_like(x[0])
				for i in range(n//2-1):
					R[X[i]<x[0]][x[0]<X[i+1]]=Rc[i]
				return R
			R_rot.interpolate(R_f)
		S_save=S
	# Check if necessary to compute the mode
	d,_=findStuff(spyp.resolvent_path+"gains/txt/",{'Re':Re,'S':S,'m':m,'St':St},distributed=False,return_distance=True)
	if not np.isclose(d,0,atol=params['atol']):
		if m_save!=m:
			comm.barrier()
			boundaryConditionsPerturbations(spyp,m)
			# For efficiency, matrices assembled once per Sts
			spyp.assembleLMatrix(m)
			m_save=m
		# Resolvent analysis
		spyp.resolvent(1,[St],Re,S,m)
	if p0 and calc_gains:
		gains_file=findStuff(spyp.resolvent_path+"gains/txt/",{"Re":Re,"S":S,"m":m,"St":St},distributed=False)
		res["gains"][S][m][St]=np.max(np.loadtxt(gains_file))
	# Approximate k as division of u
	us=spyp.readMode(source,dat)
	ux,ur,ut=ufl.split(us)
	interp_scalar(k,(ux.dx(0)/ux+ur.dx(0)/ur+ut.dx(0)/ut)/3,True)
	if calc_avgs or calc_devs:
		interp_vector(Lambda,k,m)

		# Integration area (envelope defined as 10% of abs(u))
		interp_scalar(kernel,ufl.inner(us,us))
		A=np.copy(kernel.x.array[:].real)
		# Compute max mode norm
		AM = comm.gather(np.max(A))
		if p0: AM = max(AM)
		AM = comm.scatter([AM for _ in range(comm.size)])
		# Mask everything below 10%
		kernel.x.array[A>=.1*AM] = 1
		kernel.x.array[A< .1*AM] = 0
		kernel.x.scatter_forward()
		# Total weighting surface for averaging
		A=dfx.fem.assemble_scalar(dfx.fem.form(kernel*ufl.dx)).real
		A=comm.gather(A)
		if p0: A = sum(A)

		# Compute orientation
		interp_scalar(chi,dot(Lambda,Sigma)/dot(Lambda)/dot(Sigma)*kernel)

		# Compute and communicate integrals
		avg=dfx.fem.assemble_scalar(dfx.fem.form(chi*ufl.dx)).real
		avg=comm.gather(avg)
		if p0:
			avg = sum(avg)/A
			res["avgs"][S][m][St] = avg
		avg = comm.scatter([avg for _ in range(comm.size)])
		dev=Function(spyp.TH1)
		dev=dfx.fem.assemble_scalar(dfx.fem.form((chi-avg)**2*kernel*ufl.dx)).real
		dev=comm.gather(dev)
		if p0: res["devs"][S][m][St] = np.sqrt(sum(dev)/A)

	if calc_guptas:
		# Gupta criterion
		interp_scalar(gupta,k**2*((r*U[2])**2).dx(1)/r**3-2*k*m/r**2*U[2]*U[0].dx(1)-(k*U[0].dx(1)+m*(U[2]/r).dx(1))**2/4)
		gupta.x.scatter_forward()
		mg=np.min(gupta.x.array[:].real)
		mg=comm.gather(mg)
		if p0: res["guptas"][S][m][St] = min(mg)

	# Print stuff for sanity check purposes
	if prt:
		save_str=f"_S={S}"
		if calc_avgs or calc_devs: spyp.printStuff(dir,"Sigma" +save_str,Sigma)
		save_str+=f"_m={m:d}_St={St:.4e}"
		spyp.printStuff(dir,"k"+save_str,k)
		if calc_avgs or calc_devs: spyp.printStuff(dir,"Lambda"+save_str,Lambda)
		spyp.printStuff(dir,"kernel"+save_str,kernel)
		if calc_avgs or calc_devs: spyp.printStuff(dir,"dot"+save_str,chi)
		U,_=ufl.split(spyp.Q)
		# Inertial waves
		js=jn_zeros(m,nt)
		for n in range(-nt,nt+1):
			try:
				interp_scalar(f,(m+n/abs(n)*2/js[abs(n)-1]*k*R_rot)*U[2]/r/2*np.pi)
				spyp.printStuff(dir,f"f_n={n:d}"+save_str,f)
			except ZeroDivisionError: pass
		if calc_guptas: spyp.printStuff(dir,"gupta"+save_str,gupta)
		if p0: print("A=",A,"avg=",avg,"dev=",np.sqrt(sum(dev)/A))
	# Intermediate save
	c+=1
	if p0 and c%nc==0:
		print("INTERMEDIATE SAVE",flush=True)
		for name in res.keys():
			with open(dir+name+'.pkl', 'wb') as fp: dump(res[name], fp)
#except: pass
#finally:
# Writing everything into a double contour plot
if p0:
	for name in res.keys():
		with open(dir+name+'.pkl', 'wb') as fp: dump(res[name], fp)
		from matplotlib import pyplot as plt
		fig = plt.figure(figsize=(20,10),dpi=500)
		plt.rcParams.update({'font.size': 26})
		gs = fig.add_gridspec(1, 2, wspace=0)
		ax1, ax2 = gs.subplots(sharey=True)
		# Loop of death
		resp,resn=np.empty((len(Ss),len(Sts))),np.empty((len(Ss),len(Sts)))
		for i,S in enumerate(Ss):
			for j,St in enumerate(Sts):
				resp[i,j],resn[i,j]=res[name][S][abs(ms[0])][St],res[name][S][-abs(ms[0])][St]
		# Plotting contours
		if name=='avgs':
			# Smoothing
			s=np.array([.5,.5])
			resp,resn=gaussian_filter(resp,s),gaussian_filter(resn,s)
			# Displayed level-sets
			lvls=np.linspace(0,1,11)
			ax1.contourf(  2*Sts,Ss,resn,lvls,cmap='seismic',vmin=0,vmax=1)
			c=ax2.contourf(2*Sts,Ss,resp,lvls,cmap='seismic',vmin=0,vmax=1)
		elif name=='gains':
			s=np.array([1,1])
			resp,resn=gaussian_filter(resp,s),gaussian_filter(resn,s)
			lvls=[1000,2000,4000,8000,9000,10000,12000,16000,20000,30000,50000]
			ax1.contourf(  2*Sts,Ss,resn,lvls,cmap='Reds')
			c=ax2.contourf(2*Sts,Ss,resp,lvls,cmap='Reds')
		elif name=='guptas':
			"""s=np.array([1,1])
			resp,resn=gaussian_filter(resp,s),gaussian_filter(resn,s)"""
			ax1.contourf(  2*Sts,Ss,resn,cmap='PiYG')
			c=ax2.contourf(2*Sts,Ss,resp,cmap='PiYG')
		else:
			# Smoothing
			s=np.array([.5,.5])
			resp,resn=gaussian_filter(resp,s),gaussian_filter(resn,s)
			#lvls=np.linspace(0,.5,11)
			ax1.contourf(  2*Sts,Ss,resn,cmap='Spectral')
			c=ax2.contourf(2*Sts,Ss,resp,cmap='Spectral')
		ax1.invert_xaxis()
		# Common colorbar
		fig.colorbar(c, ax=[ax1,ax2])
		ax1.set_title(r'$m=-2$')
		ax1.set_xlabel(r'$St$')
		ax1.set_ylabel(r'$S$')
		ax2.set_title(r'$m=2$')
		ax2.set_xlabel(r'$St$')
		fig.savefig(dir+name+".png")