# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, ufl, glob #source /usr/local/bin/dolfinx-complex-mode
import numpy as np
from spy import SPY, dirCreator
import dolfinx as dfx
from petsc4py import PETSc as pet
from slepc4py import SLEPc as slp
from mpi4py.MPI import COMM_WORLD as comm

p0=comm.rank==0

# Wrapper
def assembleForm(comm,form:ufl.Form,bcs:list=[],sym=False,diag=0) -> pet.Mat:
	# JIT options for speed
	form = dfx.fem.Form(form, jit_parameters={"cffi_extra_compile_args": ["-Ofast", "-march=native"],
					  						  "cffi_libraries": ["m"]})
	# Sparsity pattern
	sp = dfx.fem.create_sparsity_pattern(form._cpp_object)
	sp.assemble()
	# Create sparse matrix
	A = dfx.cpp.la.create_matrix(comm, sp)
	A.setOption(A.Option.IGNORE_ZERO_ENTRIES, 1)
	A.setOption(A.Option.SYMMETRIC,sym)
	A = dfx.fem.assemble_matrix(A,form,bcs,diag)
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

# Swirling Parallel Yaj Perturbations
class SPYP(SPY):
	def __init__(self, params:dict, datapath:str, Ref, nutf, direction_map:dict, S:float, m:int, forcingIndicator:None) -> None:
		super().__init__(params, datapath, Ref, nutf, direction_map, forcingIndicator)
		self.S=S; self.m=m
		self.save_string=f"_S={S:00.3f}_m={m:00.2f}"
		dirCreator(self.resolvent_path)

	# To be run in complex mode
	def assembleJNMatrices(self,Re:int) -> None:
		# Load baseflow
		self.loadStuff([self.S,Re],self.dat_complex_path,['S','Re'],self.q.vector)
		# Enforce no azimuthal flow in case S=0
		if self.S==0:
			_, w_dofs = self.TH.sub(0).sub(self.direction_map['th']).collapse(collapsed_dofs=True)
			self.q.x.array[w_dofs]=np.zeros(len(w_dofs))
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
		self.J = assembleForm(comm,J_form,self.bcs,diag=1)
		self.N = assembleForm(comm,N_form,self.bcs,True)
		
		if p0: print("Jacobian & Norm matrices computed !")

	# Assemble norm matrix
	def assembleMRMatrices(self):
		# Fonctions des petits et grands espaces
		u,_ = ufl.split(self.trial)
		v,_ = ufl.split(self.test)
		w = ufl.TrialFunction(self.u_space)
		z = ufl.TestFunction( self.u_space)

		# Quadrature-extensor B (m*n) reshapes forcing vector (n*1) to (m*1) and compensates the quadrature in R.
		B_form = ufl.inner(w,v)*self.r**2*self.indic*ufl.dx
		# Mass M (n*n): required to have a proper maximisation problem in a cylindrical geometry
		M_form = ufl.inner(w,z)*self.r	 *ufl.dx # Quadrature corresponds to L2 integration
		# Mass Q (m*m): norm is u^2
		Q_form = ufl.inner(u,v)*self.r	 *ufl.dx

		# Assembling matrices
		B 	   = assembleForm(comm,B_form,self.bcs)
		self.M = assembleForm(comm,M_form,sym=True)
		Q 	   = assembleForm(comm,Q_form,sym=True)
		
		if p0: print("Quadrature, Extractor & Mass matrices computed !")

		# Sizes
		m,		n 		= B.getSize()
		m_local,n_local = B.getLocalSize()

		# Temporary vectors
		tmp1, tmp2 = dfx.Function(self.TH), dfx.Function(self.TH)
		tmp3 = dfx.Function(self.u_space)

		# Resolvent operator
		class R_class:
			def mult(cls,A,x,y):
				B.mult(x,tmp1.vector)
				tmp1.x.scatter_forward()
				self.KSPs[0].solve(tmp1.vector,tmp2.vector)
				tmp2.x.scatter_forward()
				tmp2.vector.copy(y)

			def multTranspose(cls,A,x,y):
				self.KSPs[1].solve(x,tmp2.vector)
				tmp2.x.scatter_forward()
				B.multTranspose(tmp2.vector,tmp3.vector)
				tmp3.x.scatter_forward()
				tmp3.vector.copy(y)

		# Necessary for matrix-free routine
		class LHS_class:
			def mult(cls,A,x,y):
				self.R.mult(x,tmp2.vector)
				Q.mult(tmp2.vector,tmp1.vector)
				tmp1.x.scatter_forward()
				self.R.multTranspose(tmp1.vector,y)

		self.R   = pythonMatrix([[m_local,m],[n_local,n]],  R_class,comm)
		self.LHS = pythonMatrix([[n_local,n],[n_local,n]],LHS_class,comm)

	def resolvent(self,k:int,St_list):
		# Solver
		EPS = slp.EPS(); EPS.create(comm)
		for St in St_list:
			L=self.J-1j*np.pi*St*self.N # Equations (Fourier transform is -2j pi f but Strouhal is St=fD/U=2fR/U)

			self.KSPs = []
			# Useful solvers (here to put options for computing a smart R)
			for Mat in [L,L.copy().hermitianTranspose()]:
				KSP = pet.KSP().create(comm)
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
			
			if p0:
				# Conversion back into numpy 
				gains=np.sqrt(np.array([np.real(EPS.getEigenvalue(i)) for i in range(n)], dtype=np.float))
				#write gains
				dirCreator(self.resolvent_path)
				np.savetxt(self.resolvent_path+"gains"+self.save_string+f"_St={St:00.3f}.txt",gains)
				# Pretty print
				print("# of CV eigenvalues : "+str(n))
				print("# of iterations : "+str(EPS.getIterationNumber()))
				# Get a list of all the file paths with the same parameters
				fileList = glob.glob(self.resolvent_path+"*_u" +self.save_string+f"_St={St:00.3f}_n={comm.size:d}_i=*.*")
				# Iterate over the list of filepaths & remove each file
				for filePath in fileList: os.remove(filePath)

			# Write eigenvectors
			for i in range(min(n,k)):
				# Obtain forcings as eigenvectors
				fu=dfx.Function(self.u_space)
				EPS.getEigenvector(i,fu.vector)
				self.saveStuff(self.resolvent_path+"forcing/","forcing"+self.save_string+f"_St={St:00.3f}_i={i+1:d}",fu)

				# Obtain response from forcing
				q=dfx.Function(self.TH)
				self.R.mult(fu.vector,q.vector)
				u,_=q.split()
				self.saveStuff(self.resolvent_path+"response/","response"+self.save_string+f"_St={St:00.3f}_i={i+1:d}",u)
			if p0: print("Strouhal",St,"handled !")

	# Modal analysis
	def eigenvalues(self,sigma:complex,k:int) -> None:
		# Solver
		EPS = slp.EPS().create(comm)
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
		dirCreator(self.eig_path)
		# write eigenvalues
		np.savetxt(self.eig_path+"evals"+self.save_string+f"_sigma={sigma:00.3f}_n={comm.size:d}.txt",np.column_stack([vals.real, vals.imag]))
		# Write eigenvectors back in xdmf (but not too many of them)
		for i in range(min(n,3)):
			q=dfx.Function(self.TH)
			EPS.getEigenvector(i,q.vector)
			u,p = q.split()
			self.saveStuff(self.eig_path+"u/","evec_u"+self.save_string+f"_lam={vals[i]:00.3f}",u)
		if p0: print("Eigenpairs written !")