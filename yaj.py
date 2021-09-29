"""
Created on Fri Apr  9 16:05:26 2021

@author: cwang
"""


from dolfin import *
from ufl import i,j,k
import numpy as np
import os as os
import scipy.sparse as sps
import scipy.sparse.linalg as la


class yaj():
	def __init__(self,meshpath,flowmode,dnspath,import_flag):
		self.label=flowmode
		self.dnspath=dnspath
		self.private_path='doing/'
		self.resolvent_path='resolvent/'
		self.eig_path='eigenvalues/'
		self.mesh=self.LoadMesh(meshpath)
		self.Space=self.BuildFunctionSpace()

		self.bcs=self.BoundaryConditions()
		self.ru = self.InitialConditions() #main function space
		self.r = SpatialCoordinate(self.mesh)[1] 
		self.bcp = None
		self.Trial = None
		
		if import_flag:
			directory=self.dnspath+"baseflow000.xml"
			File(directory) >> self.ru.vector()
		self.Test= self.GenerateTestFunction()

		#physical parameters
		self.Re=1000
		self.U=1
		self.D=2
		self.mu=self.U*self.D/self.Re

		#control Newton solver
		self.play=1.0 #viscosity prefator
		self.minus_play=0.0 #increment of viscosity prefactor
		self.rp=0.99 #relaxation_parameter
		self.stop_it=1 #number of iteration
		self.ae=1e-12 #absolute_tolerance

		self.form=self.OperatorNonlinear()

	def LoadMesh(self,path):
		if path.split('.')[-1]=='xml':
			# Read mesh from xml file
			print ('Reading mesh in XML format from "'+path+'"...')
			return Mesh(path)
		elif path.split('.')[-1]=='msh':
			# Convert mesh from .msh to .xml using dolfin-convert
			print('Converting mesh from msh(gmsh) format to .xml')
			os.system('dolfin-convert '+path+' '+path[:-4]+'.xml')
			return Mesh(path[:-4]+'.xml')

	def BuildFunctionSpace(self):
		FE_vector=VectorElement("Lagrange",self.mesh.ufl_cell(),2)
		FE_scalar_2=FiniteElement("Lagrange",self.mesh.ufl_cell(),2)		
		FE_scalar_1=FiniteElement("Lagrange",self.mesh.ufl_cell(),1)
		FE_Space=[]
		if self.label=='incompressible':
			FE_Space.append(FE_vector) #velocity
			FE_Space.append(FE_scalar_1) #pressure
		elif self.label=='lowMach':
			FE_Space.append(FE_vector) #velocity
			FE_Space.append(FE_scalar_1) #pressure
			FE_Space.append(FE_scalar_1) #density
		return FunctionSpace(self.mesh,MixedElement(FE_Space))

	def InitialConditions(self):
		if self.label=='incompressible':
			U_init = Expression(("1.","0.","0."), degree=2)
		elif self.label=='lowMach':
			U_init = Expression(("1.","0.","1e5","1."), degree=2)
		return interpolate(U_init, self.Space)

	def GenerateTestFunction(self):
		testFunction = TestFunction(self.Space)
		if self.label=='incompressible':
			return [as_vector((testFunction[0],testFunction[1])),testFunction[2]]
		elif self.label=='lowMach':
			return [as_vector((testFunction[0],testFunction[1])),testFunction[2],testFunction[3]]

	def OperatorNonlinear(self):
		if self.label=='incompressible':
			u=as_vector((self.ru[0],self.ru[1]))
			p=self.ru[2]

			######################################################
			# Can's code for cylindrical coordinates
			# Define operator for axi-symmetric polar coordinate system
			# in x,r,th it comes [Sxx  Sxr  Sxth ]
			#                    [Srx  Srr  Srth ]
			#                    [Sthx Sthr Sthth]

			# Gradient with x[0] is x and x[1] is r
			def grad_cyl(v):
				return as_tensor([[v[0].dx(0), v[0].dx(1), 0],
								[v[1].dx(0), v[1].dx(1), 0],
								[0, 0, v[1]/self.r]])
			
			def div_cyl(v):
				c=(1./self.r)*(self.r*v[1]).dx(1) + v[0].dx(0)
				return c

			######################################################
			#for stress tensor in Compressible case (not used in current Incompressible case)
			# strain-rate tensor d = (1/2 grad(v) + grad(v)^T)
			def d_cyl(v):
				aa = grad_cyl(v)
				return (0.5*(aa + aa.T))

			# Viscous part of the stress lambda div(u)*Identity + div(2*mu*D)
			def tau_cyl(v):
				return -(2./3.)*div_cyl(v)*Identity(3) + 2.*d_cyl(v)
			######################################################
			######################################################

			self.mu=self.mu*self.play #update viscosity
			
			#mass
			F = div_cyl(u)*self.Test[1]*self.r*dx
			#momentum
			F -= inner(grad(u)*u, self.Test[0])*self.r*dx
			F -= self.mu*inner(grad_cyl(u), grad_cyl(self.Test[0]))*self.r*dx
			F -= -inner(p, div_cyl(self.Test[0]))*self.r*dx
			return F

		elif self.label=='lowMach':

			u=as_vector((self.ru[0],self.ru[1]))
			p=self.ru[2]
			rho=self.ru[3]


			####
			#not implmented yet
			####

	def Newton(self):
		ifile =0

		while (ifile<self.stop_it and self.play>0.99):
			print("viscosity prefactor:")
			print(self.play)
			solve(self.form == 0, self.ru, self.bcs, solver_parameters={"newton_solver":{'linear_solver' : 'petsc','relaxation_parameter':self.rp,"relative_tolerance":1e-12,'maximum_iterations':30,"absolute_tolerance":self.ae}})
			if self.label=='incompressible':
				#write results in private_path for a given mu
				u_r,p_r = self.ru.split()
				ll="mu"+str(np.round(self.play, decimals=2))
				File(self.dnspath+self.private_path+"u"+f"{ifile:03d}"+ll+".pvd") << u_r
				File(self.dnspath+self.private_path+"baseflow"+f"{ifile:03d}"+ll+'_'+".xml") << self.ru.vector()
				print(self.dnspath+self.private_path+"baseflow"+f"{ifile:03d}"+ll+'_'+".xml written!")
			elif self.label=='lowMach':
				pass
			
			#write result of current mu
			File(self.dnspath+"u"+".pvd") << u_r
			File(self.dnspath+"baseflow000"+".xml") << self.ru.vector()
			print(self.dnspath+"baseflow000"+".xml written!")

			ifile+=1
			self.play-=self.minus_play #increment of viscosity
			self.form=self.OperatorNonlinear()
			self.bcs=self.BoundaryConditions() #for temporal-dependant boundary condition


	def get_indices(self):
		# Collect all dirichlet boundary dof indices
		bcinds = []
		for b in self.bcp:
			bcdict = b.get_boundary_values()
			bcinds.extend(bcdict.keys())

		# total number of dofs
		N = self.Space.dim()

		# indices of free nodes
		freeinds = np.setdiff1d(range(N),bcinds,assume_unique=True).astype(np.int32)
		return freeinds

	def BoundaryConditions(self):
		eps=0.000001#DOLFIN_EPS does not work well
		#jet
		def symmetry(x, on_boundary):  #symétrie de l'écoulement stationnaire
			return x[1] < 0.0000001 and on_boundary 
		def inlet(x, on_boundary):       #entrée abscisse
			return x[0] < (-4.999999999) and on_boundary 
		def outlet(x, on_boundary):         #sortie
			return x[0] > (39.999999999) and on_boundary 
		def wall(x, on_boundary):
			return x[1] > 0.9999999999 and (-4.9999999 < x[0] < 0.000001) and on_boundary 
		def misc(x, on_boundary):      #ordonnée
			return x[1] > 9.999999999 and (0.0000001 < x[0] < 39.999999999) and on_boundary 
		# define boundary conditions for Newton/timestpper
		if self.label=='incompressible':
			ux_tanh=Expression('tanh(5*(1-x[1]))', degree=2)
			bcs_inflow_x = DirichletBC(self.Space.sub(0).sub(0), ux_tanh, inlet)
			bcs_inflow_r = DirichletBC(self.Space.sub(0).sub(1), 0, inlet)
			bcs_wall=DirichletBC(self.Space.sub(0), (0,0), wall)
			bcs_symmetry=DirichletBC(self.Space.sub(0).sub(1), 0, symmetry)
			return [bcs_inflow_x,bcs_inflow_r,bcs_wall,bcs_symmetry]
		elif self.label=='lowMach':
			pass #not implemented
			#return [bcs_square_u,bcs_square_rho,bcs_inflow_ux,bcs_inflow_uy,bcs_inflow_rho,bcs_upperandlower]

	def BoundaryConditionsPerturbations(self):
		eps=0.000001#DOLFIN_EPS does not work well
		#jet
		def symmetry(x, on_boundary):  #symétrie de l'écoulement stationnaire
			return x[1] < 0.0000001 and on_boundary 
		def inlet(x, on_boundary):       #entrée abscisse
			return x[0] < (-4.999999999) and on_boundary 
		def outlet(x, on_boundary):         #sortie
			return x[0] > (39.999999999) and on_boundary 
		def wall(x, on_boundary):
			return x[1] > 0.9999999999 and (-4.9999999 < x[0] < 0.000001) and on_boundary 
		def misc(x, on_boundary):      #upper boundary
			return x[1] > 9.999999999 and (0.0000001 < x[0] < 39.999999999) and on_boundary 
		if self.label=='incompressible':
			bcs_inflow = DirichletBC(self.Space.sub(0).sub(1), 0, inlet)
			bcs_wall=DirichletBC(self.Space.sub(0), (0,0), wall)
			bcs_symmetry=DirichletBC(self.Space.sub(0).sub(1), 0, symmetry)
			return [bcs_inflow,bcs_wall,bcs_symmetry]
		elif self.label=='lowMach':
			pass
			#return [bcs_square_rho,bcs_square_u,bcs_inflow_rho,bcs_inflow_u,bcs_upperandlower_u]

	def Resolvent(self,k,freq_list):
		parameters['linear_algebra_backend'] = 'Eigen'
		print("check base flow max and min in u:")
		print(np.max(self.ru.vector()[:]))
		print(np.min(self.ru.vector()[:]))
		self.Trial=TrialFunction(self.Space)
		#matrix A (m*m): Jacobian calculated by automatic derivative
		Aform = derivative(self.form,self.ru,self.Trial)
		Aa = assemble(Aform)
		rows, cols, values = as_backend_type(Aa).data()
		Aa = sps.csr_matrix((values, cols, rows))

		#matrix B (m*m): with matrix A form altogether the resolvent operator
		up=as_vector((self.Trial[0],self.Trial[1]))
		pp=self.Trial[2]
		B_form=inner(up,self.Test[0])*self.r*dx
		Ba = assemble(B_form)
		rows, cols, values = as_backend_type(Ba).data()
		Ba = sps.csr_matrix((values, cols, rows))

		#forcing norm Mf (m*m): here we choose ux^2+ur^2 as forcing norm
		#other userdefined norm can be used, to be added later
		Mf_form=inner(up,self.Test[0])*self.r*dx
		Mfa = assemble(Mf_form)
		rows, cols, values = as_backend_type(Mfa).data()
		Mfa = sps.csr_matrix((values, cols, rows))

		#response norm Mr (m*m): here we choose the same as forcing norm
		Mra = Mfa



		#quadrature Q (m*m): it is required to compensate the quadrature in resolvent operator R, because R=(A-i*omegaB)^(-1)
		Q_form=inner(up,self.Test[0])*self.r*dx+pp*self.Test[1]*self.r*dx
		Qa = assemble(Q_form)
		rows, cols, values = as_backend_type(Qa).data()
		Qa = sps.csr_matrix((values, cols, rows))

		self.bcp=self.BoundaryConditionsPerturbations()
		freeinds = self.get_indices()	

		#matrix P (m*n) reshapes forcing vector (n*1) to (m*1). In principal, it contains only 0 and 1 elements.
		#It can also restrict the flow regions of forcing, to be implemented later. 
		#A note for very rare case: if one wants to damp but not entirely eliminate the forcing in some regions (not the case here), one can put values between 0 and 1. In that case, the matrix I in the following is not the same as P. 
		
		index_forcing=self.Space.sub(0).dofmap().dofs() #get all index related to u
		index_forcing.sort()

		row_ind=np.intersect1d(freeinds,index_forcing) #get free index related to u
		row_ind.sort()
		m=len(self.Space.dofmap().dofs())
		n=len(row_ind)
		col_ind=np.arange(n)
		Pa=sps.csr_matrix((np.ones(n),(row_ind,col_ind)),(m,n))


		#matrix I (m*n) reshapes forcing matrix Mf (m*m) to I^T*Mf*I (n*n). The matrix I can be different from P in that very rare case remarked above.
		Ia=Pa

		A = Aa[freeinds,:][:,freeinds].tocsc()
		B = Ba[freeinds,:][:,freeinds].tocsc()
		Mf = Mfa[freeinds,:][:,freeinds].tocsc()
		Mr = Mra[freeinds,:][:,freeinds].tocsc()
		P = Pa[freeinds].tocsc()
		I = Ia[freeinds].tocsc()
		Q = Qa[freeinds,:][:,freeinds].tocsc()

		Q_shape=np.shape(Q)
		print('matrix Q size: '+str(Q_shape))
		P_shape=np.shape(P)
		print('matrix P size: '+str(P_shape))




		for freq in freq_list:
			R_inv=A-freq*1j*B
			R = la.splu(R_inv,permc_spec=3)
			# get response linear operator P^H*Q^H*R^H*Mr*R*Q*P
			def lhs(f):
				fq=Q*P*f
				re=R.solve(fq)
				x=Mr*re

				w=R.solve(x,trans='H')
				z=Q.transpose()*w
				y=P.transpose()*z			

				return y

			LHS = la.LinearOperator((min(P_shape),min(P_shape)),matvec=lhs,dtype='complex')

			# forcing linear operator is on the rhs M=I.transpose()*Mf*I
			gains ,eigenvectors = la.eigs(LHS, k=k, M=I.transpose()*Mf*I, sigma=None,  maxiter=100, tol=10-15, return_eigenvectors=True)
			
			#write forcing and response
			f0=eigenvectors[:,0]
			r0=R.solve(Q*P*f0)

			ua = Function(self.Space)

			for i in range(0,1):
				if self.label=='incompressible':
					ua.vector()[freeinds] = P*f0
					u,p  = ua.split()
					File(self.dnspath+self.resolvent_path+"forcing_u_"+str(np.round(freq, decimals=3))+".pvd") << u
					ua.vector()[freeinds] = r0
					u,p  = ua.split()
					File(self.dnspath+self.resolvent_path+"response_u_"+str(np.round(freq, decimals=3))+".pvd") << u
				if self.label=='lowMach':
					pass
			
			#write gains
			file = open(self.dnspath+self.resolvent_path+"gains.dat","w")
			for gain in gains:
				print(gain)
				file.write("%s\n" % np.real(gain))
			file.close()




	def Eigenvalues(self,sigma,k,flag_mode,savematt,loadmatt):
		parameters['linear_algebra_backend'] = 'Eigen'
		print("check base flow max and min in u:")
		print(np.max(self.ru.vector()[:]))
		print(np.min(self.ru.vector()[:]))
		
		self.Trial=TrialFunction(self.Space)

		up=as_vector((self.Trial[0],self.Trial[1]))
		pp=self.Trial[2]

		#RHS
		M_form=inner(up,self.Test[0])*self.r*dx

		Ma = assemble(M_form)

		rows, cols, values = as_backend_type(Ma).data()
		Ma = sps.csr_matrix((values, cols, rows))

		Aform = derivative(self.form,self.ru,self.Trial)
		Aa = assemble(Aform)
		rows, cols, values = as_backend_type(Aa).data()
		Aa = sps.csr_matrix((values, cols, rows))


		self.bcp = self.BoundaryConditionsPerturbations()
		freeinds = self.get_indices()

		M = Ma[freeinds,:][:,freeinds]
		A = Aa[freeinds,:][:,freeinds]


		if flag_mode==0:
			print("save matrix to file "+savematt+" and quit!")
			from scipy.io import savemat
			mdic = {"A": A, "M": M}
			savemat(savematt, mdic)
			return 0
		elif flag_mode==1:
			print("load matlab result from file "+loadmatt)
			from scipy.io import loadmat
			mdic=loadmat(loadmatt)
			vecs=mdic['V'] #if KeyError: 'V', it means the eigenvalue results are not saved into .mat
			vals=np.diag(mdic['D'])
		elif flag_mode==2:			
			print("Computing eigenvalues/vectors in Python!")
			ncv = np.max([10,2*k])
			vals, vecs = la.eigs(A, k=k, M=M, sigma=sigma, maxiter=60, tol=10e-13,ncv=ncv)
		else:
			print("Operation mode for eigenvalues is not correct. Nothing done.")
			return 0

		# only writing real parts of eigenvectors to file
		ua = Function(self.Space)
		flag_video=0 #1: export animation
		for i in range(0,1):
			ua.vector()[freeinds] = vecs[:,i]
			File(self.dnspath+self.eig_path+"evec"+str(i+1)+".xml") << ua.vector()

			if self.label=='incompressible':
				u,p  = ua.split()
				File(self.dnspath+self.eig_path+"evec_u_"+str(np.round(vals[i], decimals=3))+".pvd") << u
			if self.label=='lowMach':
				u,p,rho  = ua.split()
				File(self.dnspath+self.eig_path+"evec_rho_"+str(i+1)+".pvd") << rho
				File(self.dnspath+self.eig_path+"evec_u_"+str(i+1)+".pvd") << u
				File(self.dnspath+self.eig_path+"evec_p_"+str(i+1)+".pvd") << p
			if flag_video: # export animation
				print("Exporting video for eig "+str(i+1))
				angSteps = 20
				angList = list(2*np.pi/angSteps*np.arange(0,angSteps+1))

				angle0=np.angle(vecs[:,i])
				abs0=np.absolute(vecs[:,i])
				for k in range(0,angSteps+1):
					angle = angList[k]-angle0
					amp = abs0*np.cos(angle)
					ua.vector()[freeinds] = amp
					if self.label=='lowMach':
						u,p,rho  = ua.split()
					if self.label=='lowMach_reacting':
						u,p,rho,y  = ua.split()
					File(self.dnspath+self.eig_path+"anim_rho_"+str(i+1)+"_"+str(k)+".pvd") << rho
					File(self.dnspath+self.eig_path+"anim_u_"+str(i+1)+"_"+str(k)+".pvd") << u
					if self.label=='lowMach_reacting':
						File(self.dnspath+self.eig_path+"anim_y_"+str(i+1)+"_"+str(k)+".pvd") << y
		
		#write eigenvalues
		file = open(self.dnspath+self.eig_path+"evals.dat","w")
		for val in vals:
			print(np.real(val), np.imag(val))
			file.write("%s\n" % val)
		file.close()