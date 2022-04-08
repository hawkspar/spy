# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, ufl
import numpy as np
from spy import spy
import dolfinx as dfx
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from slepc4py import SLEPc as slp
from mpi4py.MPI import COMM_WORLD
from pdb import set_trace
from scipy.sparse import csr_matrix

# Swirling Parallel Yaj Perturbations
class spyp(spy):
	def __init__(self, datapath: str, Re: float, dM: float, S:float, m:int,meshpath: str="") -> None:
		super().__init__(datapath,Re,dM,meshpath)
		self.S=S; self.m=m
		self.save_string='_S='+f"{S:00.3f}"+'_m='+str(m)
		
	# Memoisation routine - find closest in S
	def LoadBaseflow(self,S) -> None:
		closest_file_name=self.baseflow_path+"last_baseflow_complex.dat"
		if not os.path.isdir(self.dat_complex_path): os.mkdir(self.dat_complex_path)
		file_names = [f for f in os.listdir(self.dat_complex_path) if f[-3:]=="dat"]
		d=np.infty
		for file_name in file_names:
			Sd = float(file_name[11:16]) # Take advantage of file format 
			fd = abs(S-Sd)#+abs(Re-Red)
			if fd<d: d,closest_file_name=fd,self.dat_complex_path+file_name
		viewer = pet.Viewer().createMPIIO(closest_file_name, 'r', COMM_WORLD)
		self.q.vector.load(viewer)
		self.q.vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		if COMM_WORLD.rank==0:
			print("Loaded "+closest_file_name+" as part of memoisation scheme")

	# Perturbations (really only need dofs)
	def BoundaryConditionsPerturbations(self,m) -> None:
		# Handle homogeneous boundary conditions
		homogeneous_boundaries=[(self.inlet,['x','r','th']),(self.top,['r','th'])]
		if 	     m ==0: homogeneous_boundaries.append((self.symmetry,['r','th']))
		elif abs(m)==1: homogeneous_boundaries.append((self.symmetry,['x']))
		else:		    homogeneous_boundaries.append((self.symmetry,['x','r','th']))
		self.dofps = np.empty(0,dtype=np.int32)
		for tup in homogeneous_boundaries:
			marker,directions=tup
			for direction in directions:
				dofs, _=self.ConstantBC(direction,marker)
				self.dofps=np.union1d(self.dofps,dofs)

	# To be run in complex mode
	def AssembleMatrices(self) -> None:
		# Load baseflow
		self.LoadBaseflow(self.S)
		
		# Computation of boundary condition dofs (only homogenous enforced, great for perturbations)
		self.BoundaryConditionsPerturbations(self.m)

		# Complex Jacobian of NS operator
		dform=self.LinearisedNavierStokes(self.m)
		self.J = dfx.fem.assemble_matrix(dform)
		self.J.assemble()
		self.J.zeroRowsColumnsLocal(self.dofps) # Impose homogeneous BCs

		# Forcing norm M (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		u,p = ufl.split(self.Trial)
		v,w = ufl.split(self.Test)
		norm_form=ufl.inner(u,v)*self.r**2*ufl.dx # Same multiplication process as momentum equations
		self.N = dfx.fem.assemble_matrix(norm_form)
		self.N.assemble()
		self.N.zeroRowsColumnsLocal(self.dofps,0)
		if COMM_WORLD.rank==0: print("Matrices computed !")
	
	def Resolvent(self,k:int,freq_list):
		#quadrature Q (m*m): it is required to compensate the quadrature in resolvent operator R, because R=(A-i*omegaB)^(-1)
		u,p = ufl.split(self.Trial)
		v,w = ufl.split(self.Test)
		Q_form=ufl.inner(u,v)*self.r**2*ufl.dx+ufl.inner(p,w)*self.r*ufl.dx
		Q = dfx.fem.assemble_matrix(Q_form)
		Q.assemble()

		#matrix B (m*n) reshapes forcing vector (n*1) to (m*1). In principal, it contains only 0 and 1 elements. It's essentially an extensor.
		#It can also restrict the flow regions of forcing, to be implemented later. 
		#A note for very rare case: if one wants to damp but not entirely eliminate the forcing in some regions (not the case here), one can put values between 0 and 1. In that case, the matrix I in the following is not the same as P.
		
		# Get indexes related to u
		u_space, dofs = self.Space.sub(0).collapse(collapsed_dofs=True)
		# Compute DoFs
		n_local=len(dofs)
		# Efficient creation of the B matrix
		Istart, Iend = Q.getOwnershipRange()
		m_local = Iend - Istart
		print(n_local,',',3*m_local//4)
		# Notice rows is local
		rows=csr_matrix((np.ones(n_local, dtype=bool), (dofs, np.arange(n_local))), shape=(m_local,n_local))

		# Going back to global stuff
		m=self.Space.dofmap.index_map.size_global * self.Space.dofmap.index_map_bs
		n=   u_space.dofmap.index_map.size_global *    u_space.dofmap.index_map_bs
        
		B = pet.Mat().createAIJ([[m_local,m],[n_local,n]], csr=(rows.indptr, rows.indices,rows.data), comm=COMM_WORLD)
		B.setUp()
		B.assemble()
		if COMM_WORLD.rank==0: print("Static matrices setup")
		
		# Necessary for matrix-free routine
		class LHS_class:
			def __init__(self,N,KSP,Q,B):
				self.N=N
				self.Q=Q
				self.B=B
				self.KSP=KSP
			
			def mult(self,A,x,y):
				w=pet.Vec().createMPI(m,comm=COMM_WORLD)
				z=pet.Vec().createMPI(m,comm=COMM_WORLD)
				y=pet.Vec().createMPI([n_local,n],comm=COMM_WORLD)
				self.B.mult(x,w)
				self.Q.mult(w,z)
				self.KSP.solve(z,w)
				self.N.mult(w,z)
				self.KSP.solveTranspose(z,w)
				self.Q.multTranspose(w,z)
				self.B.multTranspose(z,y)

		# Solver
		E = slp.EPS(); E.create()
		for freq in freq_list:
			L=self.J-2j*np.pi*freq*self.N # Equations
			# Useful solver
			KSP = pet.KSP().create()
			KSP.setOperators(L)
			KSP.setFromOptions()

			# Matrix free operator
			LHS=pet.Mat()
			LHS.create(comm=COMM_WORLD)
			LHS.setSizes([[n_local,n],[n_local,n]])
			LHS.setType(pet.Mat.Type.PYTHON)
			LHS.setPythonContext(LHS_class(self.N,KSP,Q,B))
			LHS.setUp()

			# Eigensolver
			E.setOperators(LHS) # Solve Rx=sigma*x (cheaper than a proper SVD)
			E.setWhichEigenpairs(E.Which.LARGEST_MAGNITUDE) # Find eigenvalues close to sigma
			E.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
			E.setTolerances(self.atol,100) # Set absolute tolerance and number of iterations
			E.setProblemType(slp.EPS.ProblemType.HEP) # Specify that A is hermitian (by construction), but M is semi-positive
			E.setFromOptions()
			if COMM_WORLD.rank==0: print("Solver ready")
			E.solve()
			n=E.getConverged()
			if n==0: continue

			# Conversion back into numpy 
			gains=np.array([E.getEigenvalue(i) for i in range(n)],dtype=np.complex)
			#write gains
			if not os.path.isdir(self.resolvent_path): os.mkdir(self.resolvent_path)
			np.savetxt(self.resolvent_path+"gains"+self.save_string+"f="+f"{freq:00.3f}"+".dat",np.abs(gains))

			# Write eigenvectors
			for i in range(min(n,k)):
				# Obtain forcings as eigenvectors
				fu=dfx.Function(self.Space.sub(0).collapse(collapsed_dofs=True)[0])
				E.getEigenvector(i,fu.vector)
				fu.x.scatter_forward()
				with XDMFFile(COMM_WORLD, self.resolvent_path+"forcing_u" +self.save_string+"_f="+f"{freq:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(fu)

				# Obtain response from forcing
				q=dfx.Function(self.Space)
				tmp=pet.Vec().createMPI(m,comm=COMM_WORLD)
				B.mult(fu.vector,q.vector)
				Q.mult(q.vector,tmp)
				KSP.solve(tmp,q.vector)
				u,p=q.split()
				with XDMFFile(COMM_WORLD, self.resolvent_path+"response_u" +self.save_string+"_f="+f"{freq:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(u)
			if COMM_WORLD.rank==0:
				print("Frequency",freq,"handled !")

	def Eigenvalues(self,sigma:complex,k:int) -> None:
		# Solver
		E = slp.EPS(COMM_WORLD); E.create()
		E.setOperators(-self.J,self.N) # Solve Ax=sigma*Mx
		E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE) # Find eigenvalues close to sigma
		E.setTarget(sigma)
		E.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
		#E.setTolerances(eps,60) # Set absolute tolerance and number of iterations
		E.setTolerances(1e-2,10)
		E.setProblemType(slp.EPS.ProblemType.PGNHEP) # Specify that A is no hermitian, but M is semi-positive
		# Spectral transform
		ST  = E.getST();    ST.setType('sinvert')
		# Krylov subspace
		KSP = ST.getKSP(); KSP.setType('preonly')
		# Preconditioner
		PC  = KSP.getPC();  PC.setType('lu')
		PC.setFactorSolverType('mumps')
		E.setFromOptions()
		E.solve()
		n=E.getConverged()
		if n==0: return
		# Conversion back into numpy 
		vals=np.array([E.getEigenvalue(i) for i in range(n)],dtype=np.complex)
		if not os.path.isdir(self.eig_path): os.mkdir(self.eig_path)
		# write eigenvalues
		np.savetxt(self.eig_path+"evals"+self.save_string+"_sigma="+f"{np.real(sigma):00.3f}"+f"{np.imag(sigma):+00.3f}"+"j.dat",np.column_stack([vals.real, vals.imag]))
		# Write eigenvectors back in xdmf (but not too many of them)
		for i in range(min(n,3)):
			q=dfx.Function(self.Space)
			E.getEigenvector(i,q.vector)
			q.x.scatter_forward()
			u,p = q.split()
			with XDMFFile(COMM_WORLD, self.eig_path+"u/evec_u_S="+f"{self.S:00.3f}"+"_m="+str(self.m)+"_lam="+f"{vals[i].real:00.3f}"+f"{vals[i].imag:+00.3f}"+"j.xdmf", "w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(u)
		if COMM_WORLD.rank==0:
			print("Eigenpairs written !")