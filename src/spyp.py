# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, ufl #source /usr/local/bin/dolfinx-complex-mode
import numpy as np
from spy import SPY
import dolfinx as dfx
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from slepc4py import SLEPc as slp
from mpi4py.MPI import COMM_WORLD

p0=COMM_WORLD.rank==0

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
		self.nutf(self.S)

		# Complex Jacobian of NS operator
		dform = self.linearisedNavierStokes(self.m)
		self.J = dfx.fem.assemble_matrix(dform)
		self.J.assemble()
		self.J.zeroRows(self.dofs) # Impose homogeneous BCs (1 on diag)

		# Forcing Norm (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		u,_ = ufl.split(self.trial)
		v,_ = ufl.split(self.test)
		norm_form = ufl.inner(u,v)*self.r**2*ufl.dx # Same multiplication process as base equations
		self.N = dfx.fem.assemble_matrix(norm_form)
		self.N.assemble()
		self.N.zeroRows(self.dofs,0) # Impose homogeneous BCs (0 on diag)
		if p0: print("Jacobian & Norm matrices computed !")

	def assembleMRMatrices(self):
		# Fonctions des petits et grands espaces
		u,_ = ufl.split(self.trial)
		v,_ = ufl.split(self.test)
		w = ufl.TrialFunction(self.u_space)
		z = ufl.TestFunction( self.u_space)

		# Quadrature-extensor B (m*n) reshapes forcing vector (n*1) to (m*1) and compensates the quadrature in R.
		B_form = ufl.inner(w,v)*self.r**2*ufl.dx
		# Extractor H (n*m) reshapes response vector (m*1) into (n*1)
		#H_form = ufl.inner(u,z)*ufl.dx
		# Mass M (n*n): required to have a proper maximisation problem in a cylindrical geometry
		M_form = ufl.inner(w,z)*self.r*ufl.dx # Quadrature corresponds to L2 integration
		O_form = ufl.inner(u,v)*self.r*ufl.dx # Quadrature corresponds to L2 integration
		# Assembling matrices
		self.B = dfx.fem.assemble_matrix(B_form); self.B.assemble(); self.B.zeroRows(self.dofs,0)
		#self.H = dfx.fem.assemble_matrix(H_form); self.H.assemble()
		O = dfx.fem.assemble_matrix(O_form); O.assemble()
		self.M = dfx.fem.assemble_matrix(M_form); self.M.assemble()
		if p0: print("Quadrature, Extractor & Mass matrices computed !")

		# Sizes
		m =      self.TH.dofmap.index_map.size_global * 	 self.TH.dofmap.index_map_bs
		n = self.u_space.dofmap.index_map.size_global * self.u_space.dofmap.index_map_bs
		n_local = self.M.local_size[0]
		Istart, Iend = self.N.getOwnershipRange()
		m_local = Iend - Istart
		
		"""print(self.H.norm())
		#self.H=self.H.transpose()
		I = pet.Mat().createAIJ([10,10],csr=(range(11),range(10),np.ones(10,dtype='bool')),comm=COMM_WORLD)
		I.setUp()
		I.assemble()
		print(I.norm())
		print(I.getValues(range(10),range(10)))"""

		# Resolvent operator
		w, z = self.N.createVecs()
		a, b = 		O.createVecs()
		class R_class:
			def mult(cls,A,x,y):
				self.B.mult(x,w)
				self.KSPs[0].solve(w,y)
				y.ghostUpdate(addv=pet.InsertMode.INSERT,
							  mode=pet.ScatterMode.FORWARD)
				#self.H.mult(z,y)

			def multTranspose(cls,A,x,y):
				#self.H.multTranspose(x,w)
				self.KSPs[1].solve(x,z)
				z.ghostUpdate(addv=pet.InsertMode.INSERT,
							  mode=pet.ScatterMode.FORWARD)
				self.B.multTranspose(z,y)

		# Necessary for matrix-free routine
		class LHS_class:
			def mult(cls,A,x,y):
				self.R.mult(x,a)
				O.mult(a,b)
				self.R.multTranspose(b,y)

		self.R=pet.Mat().create(comm=COMM_WORLD)
		self.R.setSizes([[m_local,m],[n_local,n]])
		self.R.setType(pet.Mat.Type.PYTHON)
		self.R.setPythonContext(R_class())
		self.R.setUp()
		
		self.LHS=pet.Mat().create(comm=COMM_WORLD)
		self.LHS.setSizes([[n_local,n],[n_local,n]])
		self.LHS.setType(pet.Mat.Type.PYTHON)
		self.LHS.setPythonContext(LHS_class())
		self.LHS.setUp()

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
				KSP.setTolerances(rtol=self.params['rtol'], atol=self.params['atol'], max_it=self.params['max_iter'])
				# Krylov subspace
				KSP.setType('preonly')
				# Preconditioner
				PC = KSP.getPC(); PC.setType('lu')
				PC.setFactorSolverType('mumps')
				KSP.setFromOptions()
				self.KSPs.append(KSP)

			# Eigensolver
			EPS.setOperators(self.LHS,self.M) # Solve QE^T*L^-1H*M*L^-1*QE*f=sigma^2*M*f (cheaper than a proper SVD)
			EPS.setProblemType(slp.EPS.ProblemType.GHEP) # Specify that A is hermitian (by construction), & M is semi-positive
			EPS.setWhichEigenpairs(EPS.Which.LARGEST_MAGNITUDE) # Find largest eigenvalues
			EPS.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
			EPS.setTolerances(self.params['atol'],self.params['max_iter']) # Set absolute tolerance and number of iterations
			# Spectral transform
			ST  = EPS.getST();   ST.setType('shift')
			ST.setShift(0)
			# Krylov subspace
			KSP =  ST.getKSP(); KSP.setType('preonly')
			# Preconditioner
			PC  = KSP.getPC();   PC.setType('lu')
			PC.setFactorSolverType('mumps')
			EPS.setFromOptions()
			if p0: print("Solver launch...")
			EPS.solve()
			n=EPS.getConverged()
			if n==0: continue

			# Conversion back into numpy 
			gains=np.array([EPS.getEigenvalue(i) for i in range(n)], dtype=np.complex)
			#write gains
			if not os.path.isdir(self.resolvent_path): os.mkdir(self.resolvent_path)
			np.savetxt(self.resolvent_path+"gains"+self.save_string+"_St="+f"{St:00.3f}"+".dat",np.abs(gains))

			# Write eigenvectors
			for i in range(min(n,k)):
				# Obtain forcings as eigenvectors
				fu=dfx.Function(self.u_space)
				EPS.getEigenvector(i,fu.vector)
				#fu.x.scatter_forward()
				with XDMFFile(COMM_WORLD, self.resolvent_path+"forcing_u" +self.save_string+"_St="+f"{St:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(fu)

				# Obtain response from forcing
				q=dfx.Function(self.TH)

				ids=self.dofs
				print(np.max(np.abs(fu.vector[ids])))
				w,_=self.N.createVecs()
				self.B.mult(fu.vector,w)
				print(np.max(np.abs(w[ids])))
				self.KSPs[0].solve(w,q.vector)
				print(np.max(np.abs(q.vector[ids])))
				"""self.H.mult(z,u.vector)
				print(np.max(np.abs(u.vector[ids])))
				self.R.mult(fu.vector,q.vector)"""
				u,_=q.split()
				with XDMFFile(COMM_WORLD, self.resolvent_path+"response_u"+self.save_string+"_St="+f"{St:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(u)
			
			if p0: print("Strouhal",St,"handled !")

	# Modal analysis
	def eigenvalues(self,sigma:complex,k:int) -> None:
		# Solver
		EPS = slp.EPS(COMM_WORLD).create()
		EPS.setOperators(-self.J,self.N) # Solve Ax=sigma*Mx
		EPS.setProblemType(slp.EPS.ProblemType.PGNHEP) # Specify that A is no hermitian, but M is semi-positive
		EPS.setWhichEigenpairs(EPS.Which.TARGET_MAGNITUDE) # Find eigenvalues close to sigma
		EPS.setTarget(sigma)
		EPS.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
		EPS.setTolerances(self.params['atol'],self.params['max_iter']) # Set absolute tolerance and number of iterations
		# Spectral transform
		ST  = EPS.getST();   ST.setType('sinvert')
		# Krylov subspace
		KSP =  ST.getKSP(); KSP.setType('preonly')
		# Preconditioner
		PC  = KSP.getPC();   PC.setType('lu')
		PC.setFactorSolverType('mumps')
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
			with XDMFFile(COMM_WORLD, self.eig_path+"u/evec_u_S="+f"{self.S:00.3f}"+"_m="+str(self.m)+"_lam="+f"{vals[i].real:00.3f}"+f"{vals[i].imag:+00.3f}"+"j.xdmf", "w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(u)
		if p0: print("Eigenpairs written !")