# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import numpy as np #source /usr/local/bin/dolfinx-complex-mode
import os, ufl, glob
import dolfinx as dfx
from dolfinx.fem import Function
from petsc4py import PETSc as pet
from slepc4py import SLEPc as slp
import plotly.graph_objects as go
from matplotlib import pyplot as plt
from mpi4py.MPI import COMM_WORLD as comm
from spy import SPY, dirCreator, loadStuff, saveStuff

p0=comm.rank==0

# Wrapper
def assembleForm(form:ufl.Form,bcs:list=[],sym=False,diag=0) -> pet.Mat:
	# JIT options for speed
	form = dfx.fem.form(form, jit_params={"cffi_extra_compile_args": ["-Ofast", "-march=native"],
					  					  "cffi_libraries": ["m"]})
	A = dfx.fem.petsc.assemble_matrix(form,bcs,diag) 
	A.setOption(A.Option.IGNORE_ZERO_ENTRIES, 1) # Probably useless after assemble
	A.setOption(A.Option.SYMMETRIC,sym)
	A.assemble()
	return A

# PETSc Matrix free method
def pythonMatrix(dims:list,py,comm) -> pet.Mat:
	Mat=pet.Mat().create(comm=comm)
	Mat.setSizes(dims)
	Mat.setType(pet.Mat.Type.PYTHON)
	Mat.setPythonContext(py())
	Mat.setUp()
	return Mat

# Krylov subspace
def configureKSP(KSP:pet.KSP,params:dict) -> None:
	KSP.setTolerances(rtol=params['rtol'], atol=params['atol'], max_it=params['max_iter'])
	# Krylov subspace
	KSP.setType('preonly')
	# Preconditioner
	PC = KSP.getPC(); PC.setType('lu')
	PC.setFactorSolverType('mumps')
	KSP.setFromOptions()

# Eigenvalue problem solver
def configureEPS(EPS:slp.EPS,k:int,params:dict) -> None:
	EPS.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
	EPS.setTolerances(params['atol'],params['max_iter']) # Set absolute tolerance and number of iterations
	#EPS.setTrueResidual(True)

# Swirling Parallel Yaj Perturbations
class SPYP(SPY):
	def __init__(self, params:dict, datapath:str, direction_map:dict, forcing_indicator=None) -> None:
		super().__init__(params, datapath, direction_map, forcing_indicator)

	# To be run in complex mode, assemble crucial matrices
	def assembleJNMatrices(self,m:int,stab=False,weak_bcs=lambda spy,u,p,m=0: 0) -> None:
		# Functions
		u,p = ufl.split(self.trial)
		v,_ = ufl.split(self.test)

		# Complex Jacobian of NS operator
		J_form = self.linearisedNavierStokes(weak_bcs,m,stab)
		# Forcing Norm (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		N_form = ufl.inner(u,v+stab*self.SUPG)*self.r**2*ufl.dx # Same multiplication process as base equations
		
		# Assemble matrices
		self.J = assembleForm(J_form,self.bcs,diag=1)
		self.N = assembleForm(N_form,self.bcs,not stab)

		if p0: print("Jacobian & Norm matrices computed !",flush=True)

	# Modal analysis
	def eigenvalues(self,sigma:complex,k:int,Re:int,nut:int,S:float,m:int,hot_start:bool=False) -> None:
		# Solver
		EPS = slp.EPS().create(comm)
		EPS.setOperators(-self.J,self.N) # Solve Ax=sigma*Mx
		EPS.setProblemType(slp.EPS.ProblemType.PGNHEP) # Specify that A is not hermitian, but M is semi-definite
		EPS.setWhichEigenpairs(EPS.Which.TARGET_MAGNITUDE) # Find eigenvalues close to sigma
		EPS.setTarget(sigma)
		configureEPS(EPS,k,self.params)
		if hot_start:
			q_0=Function(self.TH)
			loadStuff(self.eig_path+"q/",["Re","nut","S","m","l"],[Re,nut,S,m,sigma],q_0)
			EPS.setInitialSpace(q_0.vector)
		# Spectral transform
		ST = EPS.getST(); ST.setType('sinvert')
		# Krylov subspace
		KSP = ST.getKSP()
		configureKSP(KSP,self.params)
		EPS.setFromOptions()
		EPS.solve()
		n=EPS.getConverged()
		if n==0: return
		# Conversion back into numpy 
		vals=np.array([EPS.getEigenvalue(i) for i in range(n)],dtype=np.complex)
		dirCreator(self.eig_path)
		save_string=f"Re={Re:d}_nut={nut:d}_S={S:00.3f}_m={m:d}".replace('.',',')
		# Write eigenvalues
		np.savetxt(self.eig_path+save_string+"_sig={:.2f}".format(sigma).replace('.',',')+".txt",np.column_stack([vals.real, vals.imag]))
		# Memoisation of first eigenvector
		q=Function(self.TH)
		EPS.getEigenvector(0,q.vector)
		saveStuff(self.eig_path+"q/",save_string+"_l={:.2f}".format(vals[i]).replace('.',','),q)
		# Write eigenvectors back in xdmf (but not too many of them)
		for i in range(min(n,3)):
			EPS.getEigenvector(i,q.vector)
			u,p = q.split()
			self.printStuff(self.eig_path+"u/",save_string+"_l={:.2f}".format(vals[i]).replace('.',','),u)
		if p0: print("Eigenpairs written !",flush=True)

	# Assemble important matrices for resolvent
	def assembleMRMatrices(self,stab=False) -> None:
		# Velocity and full space functions
		u,_ = ufl.split(self.trial)
		v,_ = ufl.split(self.test)
		w = ufl.TrialFunction(self.TH0c)
		z = ufl.TestFunction( self.TH0c)

		# Quadrature-extensor B (m*n) reshapes forcing vector (n*1) to (m*1) and compensates the r-multiplication.
		B_form = ufl.inner(w,v+stab*self.SUPG)*self.r**2*self.indic*ufl.dx # Also includes forcing indicator to enforce placement
		# Mass M (n*n): required to have a proper maximisation problem in a cylindrical geometry
		M_form = ufl.inner(w,z)*self.r*ufl.dx # Quadrature corresponds to L2 integration
		# Mass Q (m*m): norm is u^2
		Q_form = ufl.inner(u,v)*self.r*ufl.dx

		# Assembling matrices
		B 	   = assembleForm(B_form,self.bcs)
		self.M = assembleForm(M_form,sym=True)
		Q 	   = assembleForm(Q_form,sym=True)

		if p0: print("Quadrature, Extractor & Mass matrices computed !",flush=True)

		# Sizes
		m,		n 		= B.getSize()
		m_local,n_local = B.getLocalSize()

		# Temporary vectors
		tmp1, tmp2 = Function(self.TH), Function(self.TH)
		tmp3 = Function(self.TH0c)

		# Resolvent operator
		class R_class:
			def mult(cls,A,x:pet.Vec,y:pet.Vec) -> None:
				B.mult(x,tmp1.vector)
				tmp1.x.scatter_forward()
				self.KSP.solve(tmp1.vector,tmp2.vector)
				tmp2.x.scatter_forward()
				tmp2.vector.copy(y)

			def multHermitian(cls,A,x:pet.Vec,y:pet.Vec) -> None:
				# Hand made solveHermitianTranspose (save extra LU factorisation)
				x.conjugate()
				self.KSP.solveTranspose(x,tmp2.vector)
				tmp2.vector.conjugate()
				tmp2.x.scatter_forward()
				B.multTranspose(tmp2.vector,tmp3.vector)
				tmp3.x.scatter_forward()
				tmp3.vector.copy(y)

		# Necessary for matrix-free routine
		class LHS_class:
			def mult(cls,A,x:pet.Vec,y:pet.Vec):
				self.R.mult(x,tmp2.vector)
				Q.mult(tmp2.vector,tmp1.vector)
				tmp1.x.scatter_forward()
				self.R.multHermitian(tmp1.vector,y)

		self.R   = pythonMatrix([[m_local,m],[n_local,n]],  R_class,comm)
		self.LHS = pythonMatrix([[n_local,n],[n_local,n]],LHS_class,comm)

	def resolvent(self,k:int,St_list,Re:int,nut:int,S:float,m:int,hot_start:bool=False) -> None:
		# Solver
		EPS = slp.EPS(); EPS.create(comm)

		# Useful solvers (here to put options for computing a smart R)
		L=self.J-1j*np.pi*St_list[0]*self.N
		self.KSP = pet.KSP().create(comm)
		self.KSP.setOperators(L)
		configureKSP(self.KSP,self.params)
		
		for St in St_list:
			L=self.J-1j*np.pi*St*self.N # Equations (Fourier transform is -2j pi f but Strouhal is St=fD/U=2fR/U)

			if hot_start:
				forcing_0=Function(self.TH0c)
				loadStuff(self.resolvent_path+"forcing/npy/",["Re","nut","S","m","St"],[Re,nut,S,m,St],forcing_0)
				EPS.setInitialSpace(forcing_0.vector)

			# Eigensolver
			EPS.setOperators(self.LHS,self.M) # Solve B^T*L^-1H*Q*L^-1*B*f=sigma^2*M*f (cheaper than a proper SVD)
			EPS.setProblemType(slp.EPS.ProblemType.GHEP) # Specify that A is hermitian (by construction), & M is semi-definite
			configureEPS(EPS,k,self.params)
			# Spectral transform (by default shift of 0)
			ST = EPS.getST()
			# Krylov subspace
			KSP = ST.getKSP()
			configureKSP(KSP,self.params)
			# Heavy lifting
			EPS.setFromOptions()
			if p0: print("Solver launch...",flush=True)
			EPS.solve()
			n=EPS.getConverged()
			if n==0: continue

			dirCreator(self.resolvent_path)
			dirCreator(self.resolvent_path+"gains/")
			dirCreator(self.resolvent_path+"forcing/")
			save_string=f"Re={Re:d}_nut={nut:d}_S={S:00.3f}_m={m:d}_St={St:00.3f}".replace('.',',')
			if p0:
				# Conversion back into numpy (we know gains to be real positive)
				gains=np.sqrt(np.array([np.real(EPS.getEigenvalue(i)) for i in range(n)], dtype=np.float))
				# Write gains
				np.savetxt(self.resolvent_path+"gains/"+save_string+".txt",gains)
				# Pretty print
				print("# of CV eigenvalues : "+str(n),flush=True)
				print("# of iterations : "+str(EPS.getIterationNumber()),flush=True)
				print("Error estimate : " +str(EPS.getErrorEstimate(0)), flush=True)
				# Get a list of all the file paths with the same parameters
				fileList = glob.glob(self.resolvent_path+"(forcing/print|forcing/npy|response/print|response/npy)"+save_string+"_i=*.*")
				# Iterate over the list of filepaths & remove each file
				for filePath in fileList: os.remove(filePath)
			# Write eigenvectors
			for i in range(min(n,k)):
				# Save on a proper compressed space
				forcing_i=Function(self.TH0c)
				# Obtain forcings as eigenvectors
				gain_i=np.sqrt(np.real(EPS.getEigenpair(i,forcing_i.vector)))
				forcing_i.x.scatter_forward()
				self.printStuff(self.resolvent_path+"forcing/print/",save_string+f"_i={i+1:d}",forcing_i)
				saveStuff(self.resolvent_path+"forcing/npy/",save_string+f"_i={i+1:d}",forcing_i)

				# Obtain response from forcing
				response_i=Function(self.TH)
				self.R.mult(forcing_i.vector,response_i.vector)
				response_i.x.scatter_forward()
				# Save on a proper compressed space
				velocity_i=Function(self.TH0c)
				# Scale response so that it is still unitary
				velocity_i.x.array[:]=response_i.x.array[self.TH_to_TH0]/gain_i
				self.printStuff(self.resolvent_path+"response/print/",save_string+f"_i={i+1:d}",velocity_i)
				saveStuff(self.resolvent_path+"response/npy/",save_string+f"_i={i+1:d}",velocity_i)

				"""expr=dfx.fem.Expression(self.div_nor(velocity_i,m),self.TH1.element.interpolation_points(),dtype=pet.ScalarType)
				div = Function(self.TH1)
				div.interpolate(expr)
				self.printStuff("./","sanity_check_div",div)"""

	def visualiseCurls(self,str,Re,nut,S,m,St,x):
		data = Function(self.TH0c)
		loadStuff(self.resolvent_path+str+"/npy/",["Re","nut","S","m","St"],[Re,nut,S,m,St],data)
		
		# Compute vorticity (curl)
		expr = dfx.fem.Expression(self.crl(data,m)[0],self.TH1.element.interpolation_points())
		crl = Function(self.TH1)
		crl.interpolate(expr)

		n = 1000
		rs = np.linspace(0,1,n)
		points = np.array([[x,r,0] for r in rs])
		bbtree = dfx.geometry.BoundingBoxTree(self.mesh, 2)
		cells, points_on_proc = [], []
		
		# Find cells whose bounding-box collide with the the points
		cell_candidates = dfx.geometry.compute_collisions(bbtree, points)
		# Choose one of the cells that contains the point
		colliding_cells = dfx.geometry.compute_colliding_cells(self.mesh, cell_candidates, points)
		for i, point in enumerate(points):
			if len(colliding_cells.links(i))>0:
				points_on_proc.append(point)
				cells.append(colliding_cells.links(i)[0])
		
		if len(points_on_proc)!=0:
			rs_on_proc = np.array(points_on_proc, dtype=np.float64)[:,1]
			crls = crl.eval(points_on_proc, cells)
		else: rs_on_proc, crls = None, None
		rs   = comm.gather(rs_on_proc, root=0)
		crls = comm.gather(crls, 	   root=0)

		# Actual plotting
		dir=self.resolvent_path+str+"/rolls/"
		dirCreator(dir)
		if p0:
			rs, crls = np.hstack([r for r in rs if r is not None]), np.hstack([crl.flatten() for crl in crls if crl is not None])
			ids=np.argsort(rs)
			crls,rs=crls[ids],rs[ids]
			thetas = np.linspace(0,2*np.pi,n//10)
			crls=np.real(np.outer(crls,np.exp(m*1j*thetas)))

			_, ax = plt.subplots(subplot_kw={"projection":'polar'})
			c=ax.contourf(thetas,rs,crls)
			ax.set_rorigin(0)

			plt.colorbar(c)
			plt.title("Rotational of "+str+" at plane r-"+r"$\theta$"+f" at x={x}")
			plt.savefig(dir+f"Re={Re:d}_nut={nut:d}_S={S:00.3f}_m={m:d}_St={St:00.3f}_x={x:00.1f}.png")
			plt.close()

	def visualiseStreaks(self,str,Re,nut,S,m,St,x):
		data = Function(self.TH0c)
		loadStuff(self.resolvent_path+str+"/npy/",    ["Re","nut","S","m","St"],[Re,nut,S,m,St],data)
		u = Function(self.TH0c)
		loadStuff(self.resolvent_path+"response/npy/",["Re","nut","S","m","St"],[Re,nut,S,m,St],u)
		
		expr = dfx.fem.Expression(data[1],self.TH1.element.interpolation_points())
		dr = Function(self.TH1)
		dr.interpolate(expr)
		expr = dfx.fem.Expression(data[2],self.TH1.element.interpolation_points())
		dt = Function(self.TH1)
		dt.interpolate(expr)
		
		expr = dfx.fem.Expression(u[0],self.TH1.element.interpolation_points())
		u = Function(self.TH1)
		u.interpolate(expr)

		n = 500
		sr,st = 50, 1
		a = 5e5
		rs = np.linspace(0,1.6,n)
		points = np.array([[x,r,0] for r in rs])
		bbtree = dfx.geometry.BoundingBoxTree(self.mesh, 2)
		cells, points_on_proc = [], []
		
		# Find cells whose bounding-box collide with the the points
		cell_candidates = dfx.geometry.compute_collisions(bbtree, points)
		# Choose one of the cells that contains the point
		colliding_cells = dfx.geometry.compute_colliding_cells(self.mesh, cell_candidates, points)
		for i, point in enumerate(points):
			if len(colliding_cells.links(i))>0:
				points_on_proc.append(point)
				cells.append(colliding_cells.links(i)[0])
		
		if len(points_on_proc)!=0:
			rs_on_proc = np.array(points_on_proc, dtype=np.float64)[:,1]
			us  =  u.eval(points_on_proc, cells)
			drs = dr.eval(points_on_proc, cells)
			dts = dt.eval(points_on_proc, cells)
		else: rs_on_proc, us, drs, dts = None, None, None, None
		rs  = comm.gather(rs_on_proc, root=0)
		us  = comm.gather(us, 	   	  root=0)
		drs = comm.gather(drs, 	   	  root=0)
		dts = comm.gather(dts, 	   	  root=0)

		# Actual plotting
		dir=self.resolvent_path+str+"/streaks/"
		dirCreator(dir)
		if p0:
			rs,  us  = np.hstack([r 		   for r  in rs  if r  is not None]), np.hstack([u.flatten()  for u  in us  if u  is not None])
			drs, dts = np.hstack([dr.flatten() for dr in drs if dr is not None]), np.hstack([dt.flatten() for dt in dts if dt is not None])
			ids=np.argsort(rs)
			rs,us,drs,dts=rs[ids],us[ids],drs[ids],dts[ids]
			thetas = np.linspace(0,2*np.pi,n//10)
			rss,thetass = rs[::sr],thetas[::st]
			drs,dts=drs[::sr],dts [::sr]
			us  = np.real(np.outer(us, np.exp(m*1j*thetas)))
			drs = np.real(np.outer(drs,np.exp(m*1j*thetass)))*a
			dts = np.real(np.outer(dts,np.exp(m*1j*thetass)))*a

			fig, ax = plt.subplots(subplot_kw={"projection":'polar'})
			fig.set_size_inches(10,10)
			plt.rcParams.update({'font.size': 20})
			fig.set_dpi(200)
			c=ax.contourf(thetas,rs,us,cmap='bwr')
			ax.quiver(thetass,rss,drs*np.cos(thetass)-dts*np.sin(thetass),drs*np.sin(thetass)+dts*np.cos(thetass))
			ax.set_rorigin(0)

			plt.colorbar(c)
			#plt.title("Visualisation of "+str+" vectors on velocity at plane r-"+r"$\theta$"+f" at x={x}")
			plt.savefig(dir+f"Re={Re:d}_nut={nut:d}_S={S:00.3f}_m={m:d}_St={St:00.3f}_x={x:00.1f}.png")
			plt.close()

	def readMode(self,str:str,Re:int,nut:int,S:float,m:int,St:float,coord=0):
		funs = Function(self.TH0c)
		loadStuff(self.resolvent_path+str+"/npy/",["Re","nut","S","m","St"],[Re,nut,S,m,St],funs)
		funs=funs.split()
		return funs[coord]

	def readCurl(self,str:str,Re:int,nut:int,S:float,m:int,St:float,coord=0):
		funs = Function(self.TH0c)
		loadStuff(self.resolvent_path+str+"/npy/",["Re","nut","S","m","St"],[Re,nut,S,m,St],funs)
		expr=dfx.fem.Expression(self.crl(funs,m)[coord],self.TH1.element.interpolation_points())
		crl = Function(self.TH1)
		crl.interpolate(expr)
		return crl

	def computeIsosurface(self,m:int,O:float,L:float,H:float,res_x:int,res_yz:int,r:float,f:Function,scale:str):
		# New regular mesh
		X,Y,Z = np.mgrid[O:L:res_x*1j, -H:H:res_yz*1j, -H:H:res_yz*1j]
		X,Y,Z = X.flatten(),Y.flatten(),Z.flatten()

		# Evaluation of projected value
		points = np.vstack((X,Y,Z)).T
		projected_points = np.vstack((X,np.sqrt(Y**2+Z**2),np.zeros_like(X))).T
		bbtree = dfx.geometry.BoundingBoxTree(self.mesh, 2)
		cells, points_on_proc, projected_points_on_proc = [], [], []
		
		# Find cells whose bounding-box collide with the the points
		cell_candidates = dfx.geometry.compute_collisions(bbtree, projected_points)
		# Choose one of the cells that contains the point
		colliding_cells = dfx.geometry.compute_colliding_cells(self.mesh, cell_candidates, projected_points)
		for i, point in enumerate(points):
			if len(colliding_cells.links(i))>0:
				points_on_proc.append(point)
				projected_points_on_proc.append(projected_points[i])
				cells.append(colliding_cells.links(i)[0])
		# Heavy lifting
		if len(points_on_proc)!=0: V = f.eval(projected_points_on_proc, cells)
		else: V = None
		# Gather data and points
		V = comm.gather(V, root=0)
		points_on_proc = comm.gather(points_on_proc, root=0)
		if p0:
			V = np.hstack([v.flatten() for v in V if v is not None])
			points = np.vstack([np.array(pts) for pts in points_on_proc if len(pts)>0])
			# Filter ghost values
			points, ids = np.unique(points, return_index=True, axis=0)
			points,V=points.T,V[ids]
			# Reorder everything to match mgrid
			ids=np.lexsort(np.flip(points,0))
			X,Y,Z=points
			V=np.real(V[ids]*np.exp(1j*m*np.arctan2(Y,Z))) # Proper azimuthal decomposition
			return go.Isosurface(x=X,y=Y,z=Z,value=V,
								 isomin=r*np.min(V),isomax=r*np.max(V),
								 colorscale=scale,
								 caps=dict(x_show=False, y_show=False, z_show=False))