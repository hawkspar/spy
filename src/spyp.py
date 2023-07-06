# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
#source /usr/local/bin/dolfinx-complex-mode
import glob 
import dolfinx as dfx
from os.path import isfile

from spy import SPY
from helpers import *
	
# Resolvent operator (L^-1B)
class R_class:
	def __init__(self,B,TH,TH0c) -> None:
		self.B=B
		self.KSP = pet.KSP().create(comm)
		self.tmp1,self.tmp2=Function(TH),Function(TH)
		self.tmp3=Function(TH0c)
	
	def setL(self,L,params):
		self.KSP.setOperators(L)
		configureKSP(self.KSP,params)

	def mult(self,A,x:pet.Vec,y:pet.Vec) -> None:
		self.B.mult(x,self.tmp1.vector)
		self.tmp1.x.scatter_forward()
		self.KSP.solve(self.tmp1.vector,self.tmp2.vector)
		self.tmp2.x.scatter_forward()
		self.tmp2.vector.copy(y)

	def multHermitian(self,A,x:pet.Vec,y:pet.Vec) -> None:
		# Hand made solveHermitianTranspose (save extra LU factorisation)
		x.conjugate()
		self.KSP.solveTranspose(x,self.tmp2.vector)
		self.tmp2.vector.conjugate()
		self.tmp2.x.scatter_forward()
		self.B.multTranspose(self.tmp2.vector,self.tmp3.vector)
		self.tmp3.x.scatter_forward()
		self.tmp3.vector.copy(y)

# Necessary for matrix-free routine (R^HQR)
class LHS_class:
	def __init__(self,R,N,TH) -> None:
		self.R,self.N=R,N
		self.tmp1,self.tmp2=Function(TH),Function(TH)

	def mult(self,A,x:pet.Vec,y:pet.Vec):
		self.R.mult(x,self.tmp2.vector)
		self.N.mult(self.tmp2.vector,self.tmp1.vector)
		self.tmp1.x.scatter_forward()
		self.R.multHermitian(self.tmp1.vector,y)

# Swirling Parallel Yaj Perturbations
class SPYP(SPY):
	def __init__(self, params:dict, datapath:str, mesh_name:str, direction_map:dict) -> None:
		super().__init__(params, datapath, mesh_name, direction_map)

	# Handle
	def interpolateBaseflow(self,spy:SPY) -> None:
		U_spy,_=spy.Q.split()
		U,	  _=self.Q.split()
		U.interpolate(U_spy)
		self.Nu.interpolate(spy.Nu)

	def readMode(self,str:str,dat:dict):
		funs = Function(self.TH0c)
		dat['i']=1
		loadStuff(self.resolvent_path+str+"/npy/",dat,funs)
		return funs

	def readCurl(self,str:str,dat:dict,coord='x'):
		funs = self.readMode(str,dat)
		crls=crl(self.r,self.direction_map['x'],self.direction_map['r'],self.direction_map['theta'],self.mesh,funs,dat['m'])[self.direction_map[coord]]
		expr=dfx.fem.Expression(crls,self.TH1.element.interpolation_points())
		crls = Function(self.TH1)
		crls.interpolate(expr)
		return crls

	# Probably better to do Fourier 2D
	def readK(self,str:str,dat:dict,coord=0):
		tup=self.readMode(str,dat).split()
		grds=grd(self.r,self.direction_map['x'],self.direction_map['r'],self.direction_map['theta'],tup[coord],dat['m'])
		expr=dfx.fem.Expression(grds[0]/tup[coord],self.TH1.element.interpolation_points())
		k = Function(self.TH1)
		k.interpolate(expr)
		return k

	# To be run in complex mode, assemble crucial matrix
	def assembleJMatrix(self,m:int) -> None:
		# Complex Jacobian of NS operator
		J_form = self.linearisedNavierStokes(m)
		self.J = assembleForm(J_form,self.bcs,diag=1)
		if p0: print("Jacobian matrix computed !",flush=True)

	def assembleNMatrix(self,indic=1) -> None:
		# Shorthands
		u, v = self.u, self.v
		# Norm (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		N_form = ufl.inner(u,v)*self.r*indic*ufl.dx # Same multiplication process as base equations		
		self.N = assembleForm(N_form,self.bcs,indic==1)
		if p0: print("Norm matrix computed !",flush=True)

	# Modal analysis
	def eigenvalues(self,sigma:complex,k:int,Re:int,S:float,m:int) -> None:
		# Solver
		EPS = slp.EPS().create(comm)
		EPS.setOperators(-self.J,self.N) # Solve Ax=sigma*Mx
		EPS.setWhichEigenpairs(EPS.Which.TARGET_MAGNITUDE) # Find eigenvalues close to sigma
		EPS.setTarget(sigma)
		configureEPS(EPS,k,self.params,slp.EPS.ProblemType.PGNHEP,True) # Specify that A is not hermitian, but M is semi-definite

		# File management shenanigans
		save_string=f"Re={Re:d}_S={S:.2f}_m={m:d}".replace('.',',')
		eig_name=self.eig_path+"values/"+save_string+f"_sig={sigma}".replace('.',',')+".txt"
		dirCreator(self.eig_path)
		dirCreator(self.eig_path+"values/")
		if isfile(eig_name):
			if p0: print("Found "+eig_name+" file, assuming it has enough eigenvalues, moving on...",flush=True)
			return
		if p0: print(f"Solver launch for sig={sigma}...",flush=True)
		EPS.solve()
		n=EPS.getConverged()
		if n==0:
			if p0: open(eig_name, mode='w').close() # Memoisation is important ! Even when there's no convergence save the effort next time
			return
		
		# Conversion back into numpy
		eigs=np.array([EPS.getEigenvalue(i) for i in range(n)],dtype=complex)
		# Write eigenvalues
		np.savetxt(eig_name,eigs)
		q=Function(self.TH)
		# Write a few eigenvectors back in xdmf
		for i in range(min(n,3)):
			EPS.getEigenvector(i,q.vector)
			u,_ = q.split()
			self.printStuff(self.eig_path+"print/",save_string+f"_l={eigs[i]}".replace('.',','),u)

	# Assemble important matrices for resolvent
	def assembleMRMatrices(self,indic_f=1,indic_q=1) -> None:
		# Velocity and full space functions
		v = self.v
		w = ufl.TrialFunction(self.TH0c)
		z = ufl.TestFunction( self.TH0c)

		# Norms - required to have a proper maximisation problem in a cylindrical geometry. Also includes indicators to enforce placement
		# Mass M (n*n): functionally same as above with additional zeroes for pressure
		M_form = ufl.inner(w,z)*self.r*indic_q*ufl.dx
		# Quadrature-extensor B (m*n) reshapes forcing vector (n*1) to (m*1)
		B_form = ufl.inner(w,v)*self.r*indic_f*ufl.dx

		# Assembling matrices
		self.M = assembleForm(M_form,sym=indic_q==1)
		B 	   = assembleForm(B_form,self.bcs)

		if p0: print("Mass & extensor matrices computed !",flush=True)

		# Sizes
		m,		n 		= B.getSize()
		m_local,n_local = B.getLocalSize()

		self.R_obj =   R_class(B,self.TH,self.TH0c)
		self.R     = pythonMatrix([[m_local,m],[n_local,n]],self.R_obj,comm)
		LHS_obj    = LHS_class(self.R,self.N,self.TH)
		self.LHS   = pythonMatrix([[n_local,n],[n_local,n]],LHS_obj,   comm)

	def resolvent(self,k:int,St_list,Re:int,S:float,m:int) -> None:
		# Solver
		EPS = slp.EPS(); EPS.create(comm)
		for St in St_list:
			# Folder creation
			dirCreator(self.resolvent_path)
			dirCreator(self.resolvent_path+"gains/")
			dirCreator(self.resolvent_path+"gains/txt/")
			dirCreator(self.resolvent_path+"forcing/")
			dirCreator(self.resolvent_path+"response/")
			save_string=f"Re={Re:d}_S={S:.1f}"
			save_string=(save_string+f"_m={m:d}_St={St:.4e}").replace('.',',')
			gains_name=self.resolvent_path+"gains/txt/"+save_string+".txt"
			# Memoisation
			if isfile(gains_name):
				if p0: print("Found "+gains_name+" file, assuming it has enough gains, moving on...",flush=True)
				continue

			# Equations
			self.R_obj.setL(self.J-2j*np.pi*St*self.N,self.params)
			# Eigensolver
			EPS.setOperators(self.LHS,self.M) # Solve B^T*L^-1H*Q*L^-1*B*f=sigma^2*M*f (cheaper than a proper SVD)
			configureEPS(EPS,k,self.params,slp.EPS.ProblemType.GHEP) # Specify that A is hermitian (by construction), & M is semi-definite
			# Heavy lifting
			if p0: print(f"Solver launch for (S,m,St)=({S:.1f},{m},{St:.4e})...",flush=True)
			EPS.solve()
			n=EPS.getConverged()
			if n==0:
				if p0: open(gains_name, mode='w').close() # Memoisation is important ! In case of 0 CV save the effort next time
				continue

			# Pretty print
			if p0:
				print("# of CV eigenvalues : "+str(n),flush=True)
				print("# of iterations : "+str(EPS.getIterationNumber()),flush=True)
				print("Error estimate : " +str(EPS.getErrorEstimate(0)), flush=True)
				# Conversion back into numpy (we know gains to be real positive)
				gains=np.array([EPS.getEigenvalue(i).real for i in range(n)], dtype=np.float)**.5
				# Write gains
				np.savetxt(gains_name,gains)
				print("Saved "+gains_name,flush=True)
				# Get a list of all the file paths with the same parameters (idea is to avoid ambiguity with past results)
				fileList = glob.glob(self.resolvent_path+"(forcing/print|forcing/npy|response/print|response/npy)"+save_string+"_i=*.*")
				# Iterate over the list of filepaths & remove each file
				for filePath in fileList: os.remove(filePath)
			
			# Write eigenvectors
			for i in range(min(n,k)):
				# Save on a proper compressed space
				forcing_i=Function(self.TH0c)
				# Obtain forcings as eigenvectors
				gain_i=EPS.getEigenpair(i,forcing_i.vector).real**.5
				forcing_i.x.scatter_forward()
				self.printStuff(self.resolvent_path+"forcing/print/","f_"+save_string+f"_i={i+1:d}",forcing_i)
				saveStuff(		self.resolvent_path+"forcing/npy/",		  save_string+f"_i={i+1:d}",forcing_i)
				# Obtain response from forcing
				response_i=Function(self.TH)
				self.R.mult(forcing_i.vector,response_i.vector)
				response_i.x.scatter_forward()
				# Save on a proper compressed space
				velocity_i=Function(self.TH0c)
				# Scale response so that it is still unitary
				velocity_i.x.array[:]=response_i.x.array[self.TH_to_TH0]/gain_i
				self.printStuff(self.resolvent_path+"response/print/","r_"+save_string+f"_i={i+1:d}",velocity_i)
				saveStuff(		self.resolvent_path+"response/npy/",	   save_string+f"_i={i+1:d}",velocity_i)

	def visualiseRPlane(self,str:str,dat:dict,x0:tuple,x1:tuple,n_x:int,n_th:int,n_t:int):
		import matplotlib.pyplot as plt

		fs = self.readMode(str,dat).split()
		Xs = np.linspace(x0[0],x1[0],n_x)
		Rs = np.linspace(x0[1],x1[1],n_x)
		XYZ = np.array([[x,r] for x,r in zip(Xs,Rs)])
		# Evaluation of projected value
		F = self.eval(fs[self.direction_map['x']], 	  XYZ)
		G = self.eval(fs[self.direction_map['r']],	  XYZ)
		H = self.eval(fs[self.direction_map['theta']],XYZ)

		# Actual plotting
		dir=self.resolvent_path+str+"/r_plane/"
		dirCreator(dir)
		if p0:
			print("Evaluation of perturbations done ! Drawing contours...",flush=True)
			th=np.linspace(0,2*np.pi,n_th,endpoint=False)
			Fs=azimuthalExtension(th,dat['m'],F,G,H,real=False)
			Xs=np.tile(Xs.reshape((n_x,1)),(1,n_th))
			Rths=np.outer(Rs,th)
			for i,t in enumerate(np.linspace(0,np.pi/4,n_t,endpoint=False)):
				Fts=[(F*np.exp(-1j*t)).real for F in Fs]
				for j,d in enumerate(self.direction_map.keys()):
					plt.contourf(Rths,Xs,Fts[j])
					plt.title(str+" in direction "+d+r" at plane x-$\theta$")
					plt.xlabel(r"$r\theta$")
					plt.ylabel(r"$x$")
					plt.savefig(dir+str+f"_Re={dat['Re']:d}_S={dat['S']:.1f}_m={dat['m']:d}_St={dat['St']:00.4e}_dir={d}_t={i}_rth".replace('.',',')+".png")
					plt.close()

	def visualiseXPlane(self,str:str,dat:dict,x:float,R:float,n_r:int,n_th:int,r_min:float):
		import matplotlib.pyplot as plt

		fs = self.readMode(str,dat).split()
		rs = np.linspace(0,R,n_r)
		XYZ = np.array([[x,r] for r in rs])
		# Evaluation of projected value
		F = self.eval(fs[self.direction_map['x']], 	  XYZ)
		G = self.eval(fs[self.direction_map['r']],	  XYZ)
		H = self.eval(fs[self.direction_map['theta']],XYZ)

		# Actual plotting
		dir=self.resolvent_path+str+"/x_plane/"
		dirCreator(dir)
		if p0:
			print("Evaluation of perturbations done ! Drawing isocontours...",flush=True)
			th=np.linspace(0,2*np.pi,n_th,endpoint=False)
			Fs=azimuthalExtension(th,dat['m'],F,G,H)

			for i,d in enumerate(self.direction_map.keys()):
				_, ax = plt.subplots(subplot_kw={"projection":'polar'})
				f=ax.contourf(th,rs,Fs[i])
				ax.set_rmin(r_min)
				ax.set_rorigin(.9*r_min)
				#ax.set_rticks([r_min,r_min+(R-r_min)/4,r_min+(R-r_min)/2,r_min+3*(R-r_min)/4,R])
				ax.set_rticks([r_min,r_min+(R-r_min)/2,R])

				# Recover direction
				plt.colorbar(f)
				plt.title(str+" in direction "+d+r" at plane r-$\theta$"+f" at x={x}")
				plt.savefig(dir+str+f"_Re={dat['Re']:d}_S={dat['S']:.1f}_m={dat['m']:d}_St={dat['St']:00.4e}_x={x:00.1f}_dir={d}".replace('.',',')+".png")
				plt.close()

	def visualiseCurls(self,str:str,dat:dict,x:float,R:float,n_r:int,n_th:int):
		import matplotlib.pyplot as plt

		C = self.readCurl(str,dat)

		rs = np.linspace(0,R,n_r)
		XYZ = np.array([[x,r] for r in rs])
		# Evaluation of projected value
		C = self.eval(C, XYZ)

		# Actual plotting
		dir=self.resolvent_path+str+"/vorticity/"
		dirCreator(dir)
		if p0:
			th=np.linspace(0,2*np.pi,n_th,endpoint=False)
			C=azimuthalExtension(th,dat['m'],C)

			_, ax = plt.subplots(subplot_kw={"projection":'polar'})
			c=ax.contourf(th,rs,C)
			ax.set_rorigin(0)

			plt.colorbar(c)
			plt.title("Rotational of "+str+r" at plane r-$\theta$"+f" at x={x}")
			plt.savefig(dir+str+f"_Re={dat['Re']:d}_S={dat['S']:.1f}_m={dat['m']:d}_St={dat['St']:00.4e}_x={x:00.1f}".replace('.',',')+".png")
			plt.close()

	def visualiseQuiver(self,str:str,dat:dict,x:float,R:float,n_r:int,n_th:int,r:int,a:int,r_min:float):
		import matplotlib.pyplot as plt

		fs = self.readMode(str,dat).split()
		u = self.readMode("response",dat).split()[self.direction_map['x']]

		rs = np.linspace(0,R,n_r)
		rs_r=rs[::r]
		XYZ   = np.array([[x,r] for r in rs])
		XYZ_r = np.array([[x,r] for r in rs_r])
		# Evaluation of projected value
		F = self.eval(fs[self.direction_map['r']],	  XYZ_r)
		G = self.eval(fs[self.direction_map['theta']],XYZ_r)
		U = self.eval(u, XYZ)

		# Actual plotting
		dir=self.resolvent_path+str+"/quiver/"
		dirCreator(dir)
		if p0:
			th = np.linspace(0,2*np.pi,n_th,endpoint=False)
			U,F,G=azimuthalExtension(th,dat['m'],U,F,G)
			F,G=F[::r,::r]*a,G[::r,::r]*a

			fig, ax = plt.subplots(subplot_kw={"projection":'polar'})
			fig.set_size_inches(10,10)
			plt.rcParams.update({'font.size': 20})
			fig.set_dpi(200)
			c=ax.contourf(th,rs,U,cmap='bwr')
			ax.quiver(th[::r],rs_r,F,G)
			ax.set_rmin(r_min)
			ax.set_rorigin(.9*r_min)
			ax.set_rticks([r_min,r_min+(R-r_min)/2,R])

			plt.colorbar(c)
			plt.title("Visualisation of "+str+r" vectors\\on velocity at plane r-$\theta$"+f" at x={x}")
			plt.savefig(dir+str+f"_Re={dat['Re']:d}_S={dat['S']:.1f}_m={dat['m']:d}_St={dat['St']:00.4e}_x={x:00.1f}".replace('.',',')+".png")
			plt.close()

	def computeIsosurfaces(self,str:str,dat:dict,XYZ:np.array,r:float,n:int,scale:str,name:str,all_dirs:bool=False) -> list:
		import plotly.graph_objects as go #pip3 install plotly
		
		X,Y,Z = XYZ
		fs = self.readMode(str,dat).split()
		# Evaluation of projected value
		XYZ_p = np.vstack((X,np.sqrt(Y**2+Z**2)))
		XYZ_e, U = self.eval(fs[self.direction_map['x']], XYZ_p.T,XYZ.T)
		if all_dirs:
			_, 	   V = self.eval(fs[self.direction_map['r']], XYZ_p.T,XYZ.T)
			_, 	   W = self.eval(fs[self.direction_map['theta']],XYZ_p.T,XYZ.T)

		if p0:
			print("Evaluation of perturbations done ! Drawing isosurfaces...",flush=True)
			X,Y,Z = XYZ_e.T
			th = np.arctan2(Z,Y)
			if all_dirs: U,V,W=azimuthalExtension(th,dat['m'],U,V,W,real=False,outer=False)
			else: 		 U=azimuthalExtension(th,dat['m'],U,real=False,outer=False)
			# Now handling time
			surfs = [[]]*(1+2*all_dirs)
			for t in np.linspace(0,np.pi/4,n,endpoint=False):
				Ut = (U*np.exp(-1j*t)).real # Time-shift
				surfs[0].append(go.Isosurface(x=X,y=Y,z=Z,value=Ut,
										   	  isomin=r*np.min(Ut),isomax=r*np.max(Ut),
										   	  colorscale=scale, name="axial "+name,
										   	  caps=dict(x_show=False, y_show=False, z_show=False),showscale=False))
				if all_dirs:
					Vt = (V*np.exp(-1j*t)).real # Moving to Cartesian referance frame at last moment
					Wt = (W*np.exp(-1j*t)).real
					surfs[1].append(go.Isosurface(x=X,y=Y,z=Z,value=Vt,
												isomin=r*np.min(Vt),isomax=r*np.max(Vt),
												colorscale=scale, name="radial "+name,
												caps=dict(x_show=False, y_show=False, z_show=False),showscale=False))
					surfs[2].append(go.Isosurface(x=X,y=Y,z=Z,value=Wt,
												isomin=r*np.min(Wt),isomax=r*np.max(Wt),
												colorscale=scale, name="azimuthal "+name,
												caps=dict(x_show=False, y_show=False, z_show=False),showscale=False))
			return surfs