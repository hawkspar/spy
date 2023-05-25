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
	def __init__(self,R,Q,TH) -> None:
		self.R,self.Q=R,Q
		self.tmp1,self.tmp2=Function(TH),Function(TH)

	def mult(self,A,x:pet.Vec,y:pet.Vec):
		self.R.mult(x,self.tmp2.vector)
		self.Q.mult(self.tmp2.vector,self.tmp1.vector)
		self.tmp1.x.scatter_forward()
		self.R.multHermitian(self.tmp1.vector,y)

# Swirling Parallel Yaj Perturbations
class SPYP(SPY):
	def __init__(self, params:dict, datapath:str, mesh_name:str, direction_map:dict) -> None:
		super().__init__(params, datapath, mesh_name, direction_map)
		dirCreator(self.baseflow_path)

	# Handle
	def interpolateBaseflow(self,spy:SPY) -> None:
		U_spy,_=spy.Q.split()
		U,	  _=self.Q.split()
		U.interpolate(U_spy)
		self.Nu.interpolate(spy.Nu)

	# To be run in complex mode, assemble crucial matrices
	def assembleJNMatrices(self,m:int) -> None:
		# Functions
		u, v = self.u, self.v

		# Complex Jacobian of NS operator
		J_form = self.linearisedNavierStokes(m)
		# Forcing Norm (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		N_form = ufl.inner(u,v)*self.r*ufl.dx # Same multiplication process as base equations
		
		# Assemble matrices
		self.J = assembleForm(J_form,self.bcs,diag=1)
		self.N = assembleForm(N_form,self.bcs,True)

		if p0: print("Jacobian & Norm matrices computed !",flush=True)

	# Modal analysis
	def eigenvalues(self,sigma:complex,k:int,Re:int,S:float,m:int) -> None:
		# Solver
		EPS = slp.EPS().create(comm)
		EPS.setOperators(-self.J,self.N) # Solve Ax=sigma*Mx
		EPS.setWhichEigenpairs(EPS.Which.TARGET_MAGNITUDE) # Find eigenvalues close to sigma
		EPS.setTarget(sigma)
		configureEPS(EPS,k,self.params,slp.EPS.ProblemType.PGNHEP,True) # Specify that A is not hermitian, but M is semi-definite

		# File management shenanigans
		save_string=f"Re={Re:d}_S={S:.1f}_m={m:d}".replace('.',',')
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
			if p0: open(eig_name, mode='w').close() # Memoisation is important ! In case of 0 CV save the effort next time
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
	def assembleMRMatrices(self,indic=1) -> None:
		# Velocity and full space functions
		u, v = self.u, self.v
		w = ufl.TrialFunction(self.TH0c)
		z = ufl.TestFunction( self.TH0c)

		# Mass Q (m*m): norm is u^2
		Q_form = ufl.inner(u,v)*self.r*ufl.dx
		# Mass M (n*n): required to have a proper maximisation problem in a cylindrical geometry
		M_form = ufl.inner(w,z)*self.r*ufl.dx # Quadrature corresponds to L2 integration
		# Quadrature-extensor B (m*n) reshapes forcing vector (n*1) to (m*1) and compensates the r-multiplication.
		B_form = ufl.inner(w,v)*self.r*indic*ufl.dx # Also includes forcing indicator to enforce placement

		# Assembling matrices
		Q 	   = assembleForm(Q_form,sym=True)
		self.M = assembleForm(M_form,sym=True)
		B 	   = assembleForm(B_form,self.bcs)

		if p0: print("Quadrature, Extractor & Mass matrices computed !",flush=True)

		# Sizes
		m,		n 		= B.getSize()
		m_local,n_local = B.getLocalSize()

		self.R_obj =   R_class(B,self.TH,self.TH0c)
		self.R     = pythonMatrix([[m_local,m],[n_local,n]],self.R_obj,comm)
		LHS_obj    = LHS_class(self.R,Q,self.TH)
		self.LHS   = pythonMatrix([[n_local,n],[n_local,n]],LHS_obj,   comm)

	def resolvent(self,k:int,St_list,Re:int,S:float,m:int) -> None:
		# Solver
		EPS = slp.EPS(); EPS.create(comm)
		for St in St_list:
			# Folder creation
			dirCreator(self.resolvent_path)
			dirCreator(self.resolvent_path+"gains/")
			dirCreator(self.resolvent_path+"forcing/")
			dirCreator(self.resolvent_path+"response/")
			save_string=f"Re={Re:d}_S={S:.1f}"
			save_string=(save_string+f"_m={m:d}_St={St}").replace('.',',')
			gains_name=self.resolvent_path+"gains/"+save_string+".txt"
			# Memoisation
			if isfile(gains_name):
				if p0: print("Found "+gains_name+" file, assuming it has enough gains, moving on...",flush=True)
				continue

			# Equations
			self.R_obj.setL(self.J-2*1j*np.pi*St*self.N,self.params)
			# Eigensolver
			EPS.setOperators(self.LHS,self.M) # Solve B^T*L^-1H*Q*L^-1*B*f=sigma^2*M*f (cheaper than a proper SVD)
			configureEPS(EPS,k,self.params,slp.EPS.ProblemType.GHEP) # Specify that A is hermitian (by construction), & M is semi-definite
			# Heavy lifting
			if p0: print(f"Solver launch for (S,m,St)=({S:.1f},{m},{St})...",flush=True)
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
				# Get a list of all the file paths with the same parameters
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
				saveStuff(self.resolvent_path+"forcing/npy/",save_string+f"_i={i+1:d}",forcing_i)
				# Obtain response from forcing
				response_i=Function(self.TH)
				self.R.mult(forcing_i.vector,response_i.vector)
				response_i.x.scatter_forward()
				# Save on a proper compressed space
				velocity_i=Function(self.TH0c)
				# Scale response so that it is still unitary
				velocity_i.x.array[:]=response_i.x.array[self.TH_to_TH0]/gain_i
				self.printStuff(self.resolvent_path+"response/print/","r_"+save_string+f"_i={i+1:d}",velocity_i)
				saveStuff(self.resolvent_path+"response/npy/",save_string+f"_i={i+1:d}",velocity_i)

	def computeIsosurfaces(self,m:int,XYZ:np.array,r:float,f:Function,n:int,scale:str,name:str) -> list:
		import plotly.graph_objects as go #pip3 install plotly
		
		X,Y,Z = XYZ
		# Evaluation of projected value
		XYZ_p = np.vstack((X,np.sqrt(Y**2+Z**2)))
		XYZ, V = self.eval(f,XYZ_p.T,XYZ.T)

		if p0:
			print("Evaluation done !",flush=True)
			X,Y,Z = XYZ.T
			V *= np.exp(1j*m*np.arctan2(Z,Y)) # Proper azimuthal decomposition
			# Now handling time
			surfs = []
			for t in np.linspace(0,np.pi/4,n,endpoint=False):
				W = (V*np.exp(-1j*t)).real # Time-shift
				surfs.append(go.Isosurface(x=X,y=Y,z=Z,value=W,
										   isomin=r*np.min(W),isomax=r*np.max(W),
										   colorscale=scale, name=name,
										   caps=dict(x_show=False, y_show=False, z_show=False),showscale=False))
			return surfs

	def readMode(self,str:str,Re:int,S:float,m:int,St:float,coord=0):
		funs = Function(self.TH0c)
		loadStuff(self.resolvent_path+str+"/npy/",{"Re":Re,"S":S,"m":m,"St":St,"i":1},funs)
		funs=funs.split()
		return funs[coord]

	def readCurl(self,str:str,Re:int,S:float,m:int,St:float,coord=0):
		funs = Function(self.TH0c)
		loadStuff(self.resolvent_path+str+"/npy/",{"Re":Re,"S":S,"m":m,"St":St,"i":1},funs)
		crls=crl(self.r,self.direction_map['x'],self.direction_map['r'],self.direction_map['th'],self.mesh,funs,m)[coord]
		expr=dfx.fem.Expression(crls,self.TH1.element.interpolation_points())
		crls = Function(self.TH1)
		crls.interpolate(expr)
		return crls

	# Probably better to do Fourier 2D
	def readK(self,str:str,Re:int,S:float,m:int,St:float,coord=0):
		funs = Function(self.TH0c)
		loadStuff(self.resolvent_path+str+"/npy/",{"Re":Re,"S":S,"m":m,"St":St,"i":1},funs)
		tup=funs.split()
		grds=grd(self.r,self.direction_map['x'],self.direction_map['r'],self.direction_map['th'],tup[coord],m)
		expr=dfx.fem.Expression(grds[0]/tup[coord],self.TH1.element.interpolation_points())
		k = Function(self.TH1)
		k.interpolate(expr)
		return k