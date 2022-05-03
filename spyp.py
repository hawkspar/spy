# coding: utf-8
"""
Created on Fri Dec 10 12:00:00 2021

@author: hawkspar
"""
import os, ufl #source /usr/local/bin/dolfinx-complex-mode
import numpy as np
from spy import spy
import dolfinx as dfx
from dolfinx.io import XDMFFile
from petsc4py import PETSc as pet
from slepc4py import SLEPc as slp
from mpi4py.MPI import COMM_WORLD
from pdb import set_trace

# Swirling Parallel Yaj Perturbations
class spyp(spy):
	def __init__(self, datapath: str, Re: float, dM: float, S:float, m:int,meshpath: str="") -> None:
		super().__init__(datapath,Re,dM,meshpath)
		self.S=S; self.m=m
		self.save_string='_S='+f"{S:00.3f}"+'_m='+str(m)
		
	# Memoisation routine - find closest in S
	def LoadStuff(self,S,last_name,path,offset,vector) -> None:
		closest_file_name=path+last_name
		if not os.path.isdir(path): os.mkdir(path)
		file_names = [f for f in os.listdir(path) if f[-3:]=="dat"]
		d=np.infty
		for file_name in file_names:
			Sd = float(file_name[offset:offset+5]) # Take advantage of file format 
			fd = abs(S-Sd)#+abs(Re-Red)
			if fd<d: d,closest_file_name=fd,path+file_name
		viewer = pet.Viewer().createMPIIO(closest_file_name, 'r', COMM_WORLD)
		vector.load(viewer)
		vector.ghostUpdate(addv=pet.InsertMode.INSERT, mode=pet.ScatterMode.FORWARD)
		# Loading eddy viscosity too
		if COMM_WORLD.rank==0:
			print("Loaded "+closest_file_name+" as part of memoisation scheme")

	def LoadBaseflow(self,S) -> None:
		self.LoadStuff(S,"last_baseflow_complex.dat",self.dat_complex_path,11,self.q.vector)
		"""
		if S==0:
			with self.q.sub(0).sub(2).vector.localForm() as zero_loc:
				zero_loc.set(0)
		"""
		self.LoadStuff(S,"last_nut.dat",			 self.nut_path,			6,self.nut.vector)
		"""		
		u,p=self.q.split()
		with XDMFFile(COMM_WORLD, "sanity_check2.xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(u)
		u,p=self.q.split()
		with XDMFFile(COMM_WORLD, "sanity_check_nut2.xdmf", "w") as xdmf:
			xdmf.write_mesh(self.mesh)
			xdmf.write_function(self.nut)
		"""

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

		# Forcing Norm (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		u,_ = ufl.split(self.Trial)
		v,_ = ufl.split(self.Test)
		norm_form=ufl.inner(u,v)*self.r**2*ufl.dx # Same multiplication process as base equations
		self.N = dfx.fem.assemble_matrix(norm_form)
		self.N.assemble()
		self.N.zeroRowsColumnsLocal(self.dofps,0)
		if COMM_WORLD.rank==0: print("Jacobian & Norm matrices computed !")
	
	def Resolvent(self,k:int,St_list):
		#quadrature Q (m*m): it is required to compensate the quadrature in resolvent operator R, because R=(A-i*omegaB)^(-1)
		u,p = ufl.split(self.Trial)
		v,w = ufl.split(self.Test)
		Q_form=ufl.inner(u,v)*self.r**2*ufl.dx+ufl.inner(p,w)*self.r*ufl.dx # Quadrature corresponds to Linearised NS degree
		Q = dfx.fem.assemble_matrix(Q_form)
		Q.assemble()

		#matrix E (m*n) reshapes forcing vector (n*1) to (m*1). In principal, it contains only 0 and 1 elements. It's essentially an extensor.
		#It can also restrict the flow regions of forcing, to be implemented later. 
		#A note for very rare case: if one wants to damp but not entirely eliminate the forcing in some regions (not the case here), one can put values between 0 and 1. In that case, the matrix I in the following is not the same as P.
		
		# Get indexes related to u
		u_space, dofs = self.Space.sub(0).collapse(collapsed_dofs=True)
		# Reformat DoFs to compute cols
		dofs = np.array(dofs,dtype='int32')
		n_local = dofs.size
		# Data is trivial
		data = np.ones(n_local,dtype='bool')
		# Parallel columns are non-trivial
		"""
		n_locals = COMM_WORLD.gather( n_local,  root=0)
		if COMM_WORLD.rank == 0:
			n_locals=np.array([0]+n_locals,dtype='int32')
			np.cumsum(n_locals, out=n_locals)
			print(n_locals)
		else:
			n_locals = np.empty(COMM_WORLD.Get_size()+1, dtype='int32')
		COMM_WORLD.Bcast(n_locals, root=0)
		cols=np.arange(n_locals[COMM_WORLD.rank],n_locals[COMM_WORLD.rank+1],dtype='int32')
		"""
		cols=np.arange(n_local,dtype='int32')
		# Efficient creation of the rows
		Istart, Iend = Q.getOwnershipRange()
		m_local = Iend - Istart
		# Notice rows is local
		rows=np.zeros(m_local+1,dtype='int32')
		np.add.at(rows, dofs+1, 1)
		np.cumsum(rows, out=rows)

		# Going back to global stuff
		m=self.Space.dofmap.index_map.size_global * self.Space.dofmap.index_map_bs
		n=   u_space.dofmap.index_map.size_global *    u_space.dofmap.index_map_bs
        
		# Efficient creation of a properly partitioned parallel B
		E = pet.Mat().createAIJ([[m_local,m],[n_local,n]],
								csr=(rows,cols,data),
								comm=COMM_WORLD)
		E.setUp()
		E.assemble()
		# Regrouping the two
		QE=Q.matMult(E)
		Q.destroy()
		E.destroy()
		QE.assemble()
		if COMM_WORLD.rank==0: print("Quadrature & Extractor matrices computed !")

		#mass Mq (m*m): it is required to have a proper maximisation problem in a cylindrical geometry 
		u,p = ufl.split(self.Trial)
		v,w = ufl.split(self.Test)
		M_form=ufl.inner(u,v)*self.r*ufl.dx+ufl.inner(p,w)*self.r*ufl.dx # Quadrature corresponds to L2 integration
		Mq = dfx.fem.assemble_matrix(M_form)
		Mq.assemble()

		#mass Mf (n*n): it is required to have a proper maximisation problem in a cylindrical geometry
		u_space, _ = self.Space.sub(0).collapse(collapsed_dofs=True)
		u = ufl.TrialFunction(u_space)
		v = ufl.TestFunction(u_space)
		M_form=ufl.inner(u,v)*self.r*ufl.dx # Quadrature corresponds to L2 integration
		Mf = dfx.fem.assemble_matrix(M_form)
		Mf.assemble()
		if COMM_WORLD.rank==0: print("Mass matrices computed !")
		
		# Necessary for matrix-free routine
		class LHS_class:
			def __init__(self,KSP,Mq,QE):
				self.Mq=Mq
				self.QE=QE
				self.KSP=KSP
			
			def mult(self,A,x,y):
				w=pet.Vec().createMPI([m_local,m],comm=COMM_WORLD)
				z=pet.Vec().createMPI([m_local,m],comm=COMM_WORLD)
				self.QE.mult(x,w)
				self.KSP.solve(w,z)
				self.Mq.mult(z,w)
				self.KSP.solveHermitianTranspose(w,z)
				self.QE.multTranspose(z,y)

		# Solver
		S = slp.EPS(); S.create()
		for St in St_list:
			L=self.J-1j*np.pi*St*self.N # Equations

			# Useful solvers (here to put options for computing a smart R)
			KSP = pet.KSP().create()
			KSP.setOperators(L)
			KSP.setFromOptions()

			# Tests
			x=self.mesh.geometry.x
			FE_vector_1=ufl.VectorElement("Lagrange",self.mesh.ufl_cell(),1,3)
			FE_vector_2=ufl.VectorElement("Lagrange",self.mesh.ufl_cell(),2,3)
			V1 = dfx.FunctionSpace(self.mesh,FE_vector_1)
			v1 = dfx.Function(V1)
			v1.vector[:]=np.repeat(10*np.exp(-.5*(x[:,0]**2+x[:,1]**2)/.2**2),3)
			V2 = dfx.FunctionSpace(self.mesh,FE_vector_2)
			v2 = dfx.Function(V2)
			v2.interpolate(v1)
			with XDMFFile(COMM_WORLD, "gaussian_input.xdmf","w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(v2)
			ft = dfx.Function(self.Space)
			ft.vector.zeroEntries()
			_, map_U = self.Space.sub(0).collapse(collapsed_dofs=True)
			ft.vector[map_U]=v2.vector
			ft.vector.assemble()
			qt = dfx.Function(self.Space)
			KSP.solve(ft.vector,qt.vector)
			ut,pt=qt.split()
			with XDMFFile(COMM_WORLD, "gaussian_test.xdmf","w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(ut)

			# Matrix free operator
			LHS=pet.Mat()
			LHS.create(comm=COMM_WORLD)
			LHS.setSizes([[n_local,n],[n_local,n]])
			LHS.setType(pet.Mat.Type.PYTHON)
			LHS.setPythonContext(LHS_class(KSP,Mq,QE))
			LHS.setUp()

			# Eigensolver
			S.setOperators(LHS,Mf) # Solve Rx=sigma*x (cheaper than a proper SVD)
			S.setWhichEigenpairs(S.Which.LARGEST_MAGNITUDE) # Find largest eigenvalues
			S.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
			S.setTolerances(self.atol,100) # Set absolute tolerance and number of iterations
			S.setProblemType(slp.EPS.ProblemType.HEP) # Specify that A is hermitian (by construction), but M is semi-positive
			S.setFromOptions()
			if COMM_WORLD.rank==0: print("Solver launch...")
			S.solve()
			n=S.getConverged()
			if n==0: continue

			# Conversion back into numpy 
			gains=np.sqrt(np.array([S.getEigenvalue(i) for i in range(n)],dtype=np.complex))
			#write gains
			if not os.path.isdir(self.resolvent_path): os.mkdir(self.resolvent_path)
			np.savetxt(self.resolvent_path+"gains"+self.save_string+"_f="+f"{St/2:00.3f}"+".dat",np.abs(gains))

			# Write eigenvectors
			for i in range(min(n,k)):
				# Obtain forcings as eigenvectors
				fu=dfx.Function(self.Space.sub(0).collapse(collapsed_dofs=True)[0])
				S.getEigenvector(i,fu.vector)
				fu.x.scatter_forward()
				with XDMFFile(COMM_WORLD, self.resolvent_path+"forcing_u" +self.save_string+"_f="+f"{St/2:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(fu)

				# Obtain response from forcing
				q=dfx.Function(self.Space)
				tmp=pet.Vec().createMPI([m_local,m],comm=COMM_WORLD)
				E.mult(fu.vector,q.vector)
				Q.mult(q.vector,tmp)
				KSP.solve(tmp,q.vector)
				u,p=q.split()
				with XDMFFile(COMM_WORLD, self.resolvent_path+"response_u" +self.save_string+"_f="+f"{St/2:00.3f}"+"_n="+f"{i+1:1d}"+".xdmf","w") as xdmf:
					xdmf.write_mesh(self.mesh)
					xdmf.write_function(u)
			if COMM_WORLD.rank==0: print("Frequency",St/2,"handled !")

	def Eigenvalues(self,sigma:complex,k:int) -> None:
		# Solver
		S = slp.EPS(COMM_WORLD); S.create()
		S.setOperators(-self.J,self.N) # Solve Ax=sigma*Mx
		S.setWhichEigenpairs(S.Which.TARGET_MAGNITUDE) # Find eigenvalues close to sigma
		S.setTarget(sigma)
		S.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
		#S.setTolerances(eps,60) # Set absolute tolerance and number of iterations
		S.setTolerances(1e-2,10)
		S.setProblemType(slp.EPS.ProblemType.PGNHEP) # Specify that A is no hermitian, but M is semi-positive
		# Spectral transform
		ST  = S.getST();    ST.setType('sinvert')
		# Krylov subspace
		KSP = ST.getKSP(); KSP.setType('preonly')
		# Preconditioner
		PC  = KSP.getPC();  PC.setType('lu')
		PC.setFactorSolverType('mumps')
		S.setFromOptions()
		S.solve()
		n=S.getConverged()
		if n==0: return
		# Conversion back into numpy 
		vals=np.array([S.getEigenvalue(i) for i in range(n)],dtype=np.complex)
		if not os.path.isdir(self.eig_path): os.mkdir(self.eig_path)
		# write eigenvalues
		np.savetxt(self.eig_path+"evals"+self.save_string+"_sigma="+f"{np.real(sigma):00.3f}"+f"{np.imag(sigma):+00.3f}"+"j.dat",np.column_stack([vals.real, vals.imag]))
		# Write eigenvectors back in xdmf (but not too many of them)
		for i in range(min(n,3)):
			q=dfx.Function(self.Space)
			S.getEigenvector(i,q.vector)
			q.x.scatter_forward()
			u,p = q.split()
			with XDMFFile(COMM_WORLD, self.eig_path+"u/evec_u_S="+f"{self.S:00.3f}"+"_m="+str(self.m)+"_lam="+f"{vals[i].real:00.3f}"+f"{vals[i].imag:+00.3f}"+"j.xdmf", "w") as xdmf:
				xdmf.write_mesh(self.mesh)
				xdmf.write_function(u)
		if COMM_WORLD.rank==0:
			print("Eigenpairs written !")