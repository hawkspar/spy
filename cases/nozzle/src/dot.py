# coding: utf-8
"""
Created on Tue Aug  8 17:27:00 2023

@author: hawkspar
"""
from sys import path
from os.path import isfile
from pickle import dump, load
from itertools import product
from scipy.special import jn_zeros
from scipy.ndimage import gaussian_filter

path.append('/home/shared/cases/nozzle/src/')

from setup import *
from spyp import SPYP
from helpers import j, dirCreator, findStuff

# Parameter space
Ss =np.linspace(0,1,26)
Sts=np.unique(np.hstack([np.linspace(0,1,26),np.linspace(0,.02,6)]))
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
# Number of interial waves
nt=10

avgs,devs,gains,guptas={},{},{},{}

#calcs=[("avgs",avgs),("devs",devs),("gains",gains),("guptas",guptas)]
calcs=[("guptas",guptas)]

# Triggers computations
calc_avgs,calc_devs,calc_gains,calc_guptas=False,False,False,False
for name,_ in calcs:
	calc_avgs=name=="avgs"
	calc_devs=name=="devs"
	calc_gains=name=="gains"
	calc_guptas=name=="guptas"

# Initialising saving structure
if p0:
	for name,dic in calcs:
		if isfile(dir+name+".pkl"):
			with open(dir+name+'.pkl', 'rb') as fp: dic=load(fp)
		for S in Ss:
			if not str(S) in dic.keys(): dic[str(S)]={}
			for m in ms:
				if not str(m) in dic[str(S)].keys(): dic[str(S)][str(m)]={}
S_save,m_save=-np.inf,-np.inf

FS_v = dfx.fem.FunctionSpace(spyp.mesh,ufl.VectorElement("CG",spyp.mesh.ufl_cell(),2))
Sigma,Lambda = Function(FS_v),Function(FS_v)
k,kernel,chi,f,gupta = Function(spyp.TH1),Function(spyp.TH1),Function(spyp.TH1),Function(spyp.TH1),Function(spyp.TH1)
# Handle
r=spyp.r

# Handler
def dot(u,v=None):
	if v is None: return ufl.sqrt(ufl.real(ufl.dot(u,u))) # compute ||u||
	return ufl.sqrt(ufl.real(ufl.dot(u,v))**2) # compute |<u,v>|
def interp_scalar(v,v1):
	v.interpolate(dfx.fem.Expression(v1,spyp.TH1.element.interpolation_points()))
	v.x.array[:]=np.nan_to_num(v.x.array[:])
def interp_vector(v,v1,v2):
	v.interpolate(dfx.fem.Expression(ufl.as_vector([v1,v2]),FS_v.element.interpolation_points()))
	v.x.array[:]=np.nan_to_num(v.x.array[:])

#try:
c=0
for dat in dats:
	S,m,St=dat['S'],dat['m'],dat['St']
	save_str="_S="+str(S)+f"_m={m:d}_St={St:.3f}"
	# Memoisation
	cont=True
	if p0:
		try:
			for _,dic in calcs: dic[str(S)][str(m)][str(St)]
			print("Found data in dictionary, moving on...",flush=True)
		except KeyError: cont=False
	cont = comm.scatter([cont for _ in range(comm.size)])
	if cont: continue
	# Check if necessary to compute the mode
	d,_=findStuff(spyp.resolvent_path+"gains/txt/",{'Re':Re,'S':S,'m':m,'St':St},distributed=False,return_distance=True)
	if not np.isclose(d,0,atol=params['atol']) and m_save!=m:
		comm.barrier()
		boundaryConditionsPerturbations(spyp,m)
		# For efficiency, matrices assembled once per Sts
		spyp.assembleLMatrix(m)
		m_save=m
	# Load only if it changed from last computation
	if S_save!=S:
		comm.barrier()
		spyb.loadBaseflow(Re,S) # Sometimes unecessary to load nut
		spyp.interpolateBaseflow(spyb)
		if calc_avgs or calc_devs:
			U,_=ufl.split(spyp.Q)
			interp_vector(Sigma,U[0].dx(1),(U[2]/r).dx(1))
		S_save=S
	# Resolvent analysis
	spyp.resolvent(1,[St],Re,S,m)
	if p0 and calc_gains:
		gains_file=findStuff(spyp.resolvent_path+"gains/txt/",{"Re":Re,"S":S,"m":m,"St":St},distributed=False)
		try: 	gains[str(S)][str(m)][str(St)]=np.loadtxt(gains_file)[0]
		except: gains[str(S)][str(m)][str(St)]=np.loadtxt(gains_file)
	# Approximate k as division of u
	us=spyp.readMode(source,dat)
	ux,ur,ut=ufl.split(us)
	interp_scalar(k,ufl.real((ux.dx(0)/ux+ur.dx(0)/ur+ut.dx(0)/ut)/j(spyp.mesh)/3))
	if calc_avgs or calc_devs:
		interp_vector(Lambda,k,m)

		# Integration area (envelope defined as 10% of abs(u))
		interp_scalar(kernel,ufl.real(ux)**2+ufl.imag(ux)**2+\
							ufl.real(ur)**2+ufl.imag(ur)**2+\
							ufl.real(ut)**2+ufl.imag(ut)**2)
		A=np.copy(np.real(kernel.x.array[:]))
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
			avgs[str(S)][str(m)][str(St)] = avg
		avg = comm.scatter([avg for _ in range(comm.size)])
		dev=Function(spyp.TH1)
		dev=dfx.fem.assemble_scalar(dfx.fem.form((chi-avg)**2*kernel*ufl.dx)).real
		dev=comm.gather(dev)
		if p0: devs[str(S)][str(m)][str(St)] = np.sqrt(sum(dev)/A)

	if calc_guptas:
		# Gupta criterion
		interp_scalar(gupta,k**2*((r*U[2])**2).dx(1)/r**3-2*k*m/r**2*U[2]*U[0].dx(1)-(k*U[0].dx(1)+m*(U[2]/r).dx(1))**2/4)
		gupta.scatter_forward()
		mg=np.min(gupta.x.array)
		mg=comm.gather(mg)
		if p0: devs[str(S)][str(m)][str(St)] = min(mg)

	# Print stuff for sanity check purposes
	if prt:
		save_str=f"_S={S}"
		spyp.printStuff(dir,"Sigma" +save_str,Sigma)
		save_str+=f"_m={m:d}_St={St:.4e}"
		spyp.printStuff(dir,"Lambda"+save_str,Lambda)
		spyp.printStuff(dir,"kernel"+save_str,kernel)
		spyp.printStuff(dir,"chi"+save_str,chi)
		U,_=ufl.split(spyp.Q)
		# Inertial waves
		js=jn_zeros(m,nt)
		for n in range(-nt,nt+1):
			try:
				interp_scalar(f,(m+n/abs(n)*2/js[abs(n)-1]*k*r)*U[2]/r/2*np.pi)
				spyp.printStuff(dir,f"f_n={n:d}"+save_str,f)
			except ZeroDivisionError: pass
		spyp.printStuff(dir,"gupta"+save_str,gupta)
		if p0: print("A=",A,"avg=",avg,"dev=",np.sqrt(sum(dev)/A))
	# Intermediate save
	c+=1
	if p0 and c%nc==0:
		print("INTERMEDIATE SAVE",flush=True)
		for name,res in [('avgs',avgs),('devs',devs),('gains',gains)]:
			with open(dir+name+'.pkl', 'wb') as fp: dump(res, fp)
#except: pass
#finally:
# Writing everything into a double contour plot
if p0:
	for name,res in calcs:
		with open(dir+name+'.pkl', 'wb') as fp: dump(res, fp)
		from matplotlib import pyplot as plt
		fig = plt.figure(figsize=(20,10),dpi=500)
		plt.rcParams.update({'font.size': 26})
		gs = fig.add_gridspec(1, 2, wspace=0)
		ax1, ax2 = gs.subplots(sharey=True)
		# Loop of death
		resp,resn=np.empty((len(Ss),len(Sts))),np.empty((len(Ss),len(Sts)))
		for i,S in enumerate(Ss):
			for j,St in enumerate(Sts):
				resp[i,j],resn[i,j]=res[str(S)][str(abs(ms[0]))][str(St)],res[str(S)]['-'+str(abs(ms[0]))][str(St)]
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
			resp,resn=gaussian_filter(resp,s),gaussian_filter(resn,s)
			lvls=[1000,2000,4000,8000,9000,10000,12000,16000,20000,30000,50000]"""
			ax1.contourf(  2*Sts,Ss,resn,cmap='PiYG',vmin=-1,vmax=1)
			c=ax2.contourf(2*Sts,Ss,resp,cmap='PiYG',vmin=-1,vmax=1)
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