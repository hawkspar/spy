# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, ufl, glob #source /usr/local/bin/dolfinx-complex-mode
import numpy as np
from spy import SPY
import dolfinx as dfx
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from slepc4py import SLEPc as slp
from mpi4py.MPI import COMM_WORLD as comm

p0=comm.rank==0

# Wrapper
def assembleForm(form):
	jit_parameters = {"cffi_extra_compile_args": ["-Ofast", "-march=native"],
					  "cffi_libraries": ["m"]}
	# JIT options for speed
	form = dfx.fem.Form(form, jit_parameters=jit_parameters)
	# Sparsity pattern
	sp = dfx.fem.create_sparsity_pattern(form._cpp_object)
	sp.assemble()
	# Create sparse matrix
	A = dfx.cpp.la.create_matrix(comm, sp)
	A.setOption(A.Option.IGNORE_ZERO_ENTRIES, 1)
	A = dfx.fem.assemble_matrix(A,form)
	A.assemble()
	return A

# PETSc Matrix free method
def pythonMatrix(dims:list,py) -> pet.Mat:
	Mat=pet.Mat().create(comm=comm)
	Mat.setSizes(dims)
	Mat.setType(pet.Mat.Type.PYTHON)
	Mat.setPythonContext(py())
	Mat.setUp()
	return Mat

# Krylov subspace
def configureKSP(KSP:pet.KSP,params:dict):
	KSP.setTolerances(rtol=params['rtol'], atol=params['atol'], max_it=params['max_iter'])
	# Krylov subspace
	KSP.setType('preonly')
	# Preconditioner
	PC = KSP.getPC(); PC.setType('lu')
	PC.setFactorSolverType('mumps')
	KSP.setFromOptions()

# Eigenvalue problem solver
def configureEPS(EPS:slp.EPS,k:int,params:dict):
	EPS.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
	EPS.setTolerances(params['atol'],params['max_iter']) # Set absolute tolerance and number of iterations

# Swirling Parallel Yaj Perturbations
class SPYP(SPY):
	def __init__(self, params:dict, datapath:str, Ref, nutf, direction_map:dict, S:float, m:int) -> None:
		super().__init__(params, datapath, Ref, nutf, direction_map)
		self.S=S; self.m=m
		self.save_string='_S='+f"{S:00.3f}"+'_m='+str(m)

	# To be run in complex mode
	def assembleJNMatrices(self) -> None:
		# Load baseflow
		self.loadStuff(self.S,"last_baseflow.dat",self.dat_complex_path,11,self.q.vector)
		# Enforce no azimuthal flow in case S=0
		if self.S==0:
			_, w_dofs = self.TH.sub(0).sub(2).collapse(collapsed_dofs=True)
			self.q.vector[w_dofs]=np.zeros(len(w_dofs))
		# Load turbulent viscosity
		self.nutf(self.S)

		# Functions
		u,_ = ufl.split(self.trial)
		v,_ = ufl.split(self.test)

		# Complex Jacobian of NS operator
		J_form = self.linearisedNavierStokes(self.m)
		# Forcing Norm (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		N_form = ufl.inner(u,v)*self.r**2*ufl.dx # Same multiplication process as base equations

		# Assemble matrices
		self.J = assembleForm(J_form)
		self.J.zeroRowsColumns(self.dofs) # Impose homogeneous BCs (1 on diag)
		self.N = assembleForm(N_form)
		self.N.zeroRowsColumns(self.dofs,0) # Impose homogeneous BCs (0 on diag)
		if p0: print("Jacobian & Norm matrices computed !")

	# Assemble norm matrix
	def assembleMRMatrices(self):
		# Fonctions des petits et grands espaces
		u,_ = ufl.split(self.trial)
		v,_ = ufl.split(self.test)
		w = ufl.TrialFunction(self.u_space)
		z = ufl.TestFunction( self.u_space)

		# Quadrature-extensor B (m*n) reshapes forcing vector (n*1) to (m*1) and compensates the quadrature in R.
		B_form = ufl.inner(w,v)*self.r**2*ufl.dx
		# Mass M (n*n): required to have a proper maximisation problem in a cylindrical geometry
		M_form = ufl.inner(w,z)*self.r*ufl.dx # Quadrature corresponds to L2 integration
		# Mass Q (m*m): norm is u^2
		Q_form = ufl.inner(u,v)*self.r*ufl.dx

		# Assembling matrices
		B = assembleForm(B_form)
		B.zeroRowsColumns(self.dofs,0)
		self.M = assembleForm(M_form)
		Q 	   = assembleForm(Q_form)
		if p0: print("Quadrature, Extractor & Mass matrices computed !")

		# Sizes
		m,		n 		= B.getSize()
		m_local,n_local = B.getLocalSize()

		# Temporary vectors
		tmp1, tmp2 = Q.getVecs()

		# Resolvent operator
		class R_class:
			def mult(cls,A,x,y):
				B.mult(x,tmp1)
				self.KSPs[0].solve(tmp1,y)

			def multTranspose(cls,A,x,y):
				self.KSPs[1].solve(x,tmp2)
				B.multTranspose(tmp2,y)

		# Necessary for matrix-free routine
		class LHS_class:
			def mult(cls,A,x,y):
				self.R.mult(x,tmp2)
				Q.mult(tmp2,tmp1)
				self.R.multTranspose(tmp1,y)

		self.R   = pythonMatrix([[m_local,m],[n_local,n]],  R_class)
		self.LHS = pythonMatrix([[n_local,n],[n_local,n]],LHS_class)

	def resolvent(self,k:int,St_list):
		# Solver
		EPS = slp.EPS(); EPS.create()
		for St in St_list:
			L=self.J-1j*np.pi*St*self.N # Equations (Fourier transform is -2j pi f but Strouhal is St=fD/U=2fR/U)

			self.KSPs = []
			# Useful solvers (here to put options for computing a smart R)
			for Mat in [L,L.copy().hermitianTranspose()]:
				KSP = pet.KSP().create()
				KSP.setOperators(Mat)
				configureKSP(KSP,self.params)
				self.KSPs.append(KSP)

			# Eigensolver
			EPS.setOperators(self.LHS,self.M) # Solve QE^T*L^-1H*M*L^-1*QE*f=sigma^2*M*f (cheaper than a proper SVD)
			EPS.setProblemType(slp.EPS.ProblemType.GHEP) # Specify that A is hermitian (by construction), & M is semi-definite
			configureEPS(EPS,k,self.params)
			# Spectral transform (maybe unnecessary ?)
			ST = EPS.getST(); ST.setType('shift'); ST.setShift(0)
			# Krylov subspace
			KSP = ST.getKSP()
			configureKSP(KSP,self.params)
			EPS.setFromOptions()
			if p0: print("Solver launch...")
			EPS.solve()
			n=EPS.getConverged()
			if n==0: continue

			# Conversion back into numpy 
			gains=np.sqrt(np.array([np.real(EPS.getEigenvalue(i)) for i in range(n)], dtype=np.float))
			#write gains
			if not os.path.isdir(self.resolvent_path): os.mkdir(self.resolvent_path)
			np.savetxt(self.resolvent_path+"gains"+self.save_string+"_St="+f"{St:00.3f}"+".dat",np.abs(gains))
			
			if p0:
				print("# of CV eigenvalues : "+str(n))
				print("# of iterations : "+str(EPS.getIterationNumber()))
				# Get a list of all the file paths with the same parameters
				fileList = glob.glob(self.resolvent_path+"*_u" +self.save_string+"_St="+f"{St:00.3f}"+"_n=*.*")
				# Iterate over the list of filepaths & remove each file
				for filePath in fileList: os.remove(filePath)

			# Write eigenvectors
			for i in range(min(n,k)):
				# Obtain forcings as eigenvectors
				fu=dfx.Function(self.u_space)
				EPS.getEigenvector(i,fu.vector)
				with XDMFFile(comm, self.resolvent_path+"forcing_u" +self.save_string+"_St="+f"{St:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(fu)

				# Obtain response from forcing
				q=dfx.Function(self.TH)
				self.R.mult(fu.vector,q.vector)
				u,_=q.split()
				with XDMFFile(comm, self.resolvent_path+"response_u"+self.save_string+"_St="+f"{St:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(u)
			if p0: print("Strouhal",St,"handled !")

	# Modal analysis
	def eigenvalues(self,sigma:complex,k:int) -> None:
		# Solver
		EPS = slp.EPS(comm).create()
		EPS.setOperators(-self.J,self.N) # Solve Ax=sigma*Mx
		EPS.setProblemType(slp.EPS.ProblemType.PGNHEP) # Specify that A is not hermitian, but M is semi-definite
		EPS.setWhichEigenpairs(EPS.Which.TARGET_MAGNITUDE) # Find eigenvalues close to sigma
		EPS.setTarget(sigma)
		configureEPS(EPS,k,self.params)
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
		if not os.path.isdir(self.eig_path): os.mkdir(self.eig_path)
		# write eigenvalues
		np.savetxt(self.eig_path+"evals"+self.save_string+"_sigma="+f"{np.real(sigma):00.3f}"+f"{np.imag(sigma):+00.3f}"+"j.dat",np.column_stack([vals.real, vals.imag]))
		# Write eigenvectors back in xdmf (but not too many of them)
		for i in range(min(n,3)):
			q=dfx.Function(self.TH)
			EPS.getEigenvector(i,q.vector)
			q.x.scatter_forward()
			u,p = q.split()
			with XDMFFile(comm, self.eig_path+"u/evec_u_S="+f"{self.S:00.3f}"+"_m="+str(self.m)+"_lam="+f"{vals[i].real:00.3f}"+f"{vals[i].imag:+00.3f}"+"j.xdmf", "w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(u)
		if p0: print("Eigenpairs written !")