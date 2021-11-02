# coding: utf-8
"""
Created on Wed Oct 13 13:50:00 2021

@author: hawkspar
"""
from dolfin import *
import numpy as np
import os as os
import scipy.sparse as sps
import scipy.sparse.linalg as la
from ufl.coefficient import Coefficient
#from pdb import set_trace

rp =.99 #relaxation_parameter
ae =1e-9 #absolute_tolerance
eps=1e-12 #DOLFIN_EPS does not work well

def extract_up(q,shift=0): return as_vector((q[shift],q[shift+1],q[shift+2])),q[shift+3]

def load_mesh(path):
	if path.split('.')[-1]=='xml':
		# Read mesh from xml file
		print ('Reading mesh in XML format from "'+path+'"...')
		return Mesh(path)
	elif path.split('.')[-1]=='msh':
		# Convert mesh from .msh to .xml using dolfin-convert
		print('Converting mesh from msh(gmsh) format to .xml')
		os.system('dolfin-convert '+path+' '+path[:-4]+'.xml')
		return Mesh(path[:-4]+'.xml')
	
# Jet geometry
def boundary_geometry():
	def symmetry(x, on_boundary):  #symmétrie de l'écoulement stationnaire
		return x[1] < 	  eps and on_boundary 
	def inlet(	 x, on_boundary):     #entrée abscisse
		return x[0] <     eps and on_boundary 
	def outlet(	 x, on_boundary):    #sortie
		return x[0] > 120-eps and on_boundary 
	def misc(	 x, on_boundary):      #upper boundary
		return x[1] >  60-eps and on_boundary
	return symmetry,inlet,outlet,misc

# Gradient with x[0] is x, x[1] is r, x[2] is theta
def grad_re(v,r):
	return as_tensor([[v[0].dx(0), v[0].dx(1),  0],
				 	  [v[1].dx(0), v[1].dx(1), -v[2]/r],
					  [v[2].dx(0), v[2].dx(1),  v[1]/r]])

def grad_im(v,r,m):
	return m*as_tensor([[0, 0, v[0]],
						[0, 0, v[1]],
						[0, 0, v[2]]])/r

def grad(v,r,m): return grad_re(v,r)+1j*grad_im(v,r,m)

def div_re(v,r):   return v[0].dx(0) + (r*v[1]).dx(1)/r

def div_im(v,r,m): return m*v[2]/r

def div(v,r,m): return div_re(v,r)+1j*div_im(v,r,m)
class yaj():
	def __init__(self,meshpath,dnspath,m,Re,S,n_S):
		#control Newton solver
		self.nu	 =1. #viscosity prefator
		self.n_nu=1 #number of visocity iterations
		self.S   =S #swirl amplitude relative to main flow
		self.n_S =n_S #number of swirl iterations
		self.m  =m # azimuthal decomposition

		# Fundamental flow type
		self.dnspath=dnspath
		self.private_path  	='doing/'
		self.resolvent_path	='resolvent/'
		self.eig_path		='eigenvalues/'
		#Re_string		='_Re='+str(Re)+'_'
		self.baseflow_string='_S='+f"{S:00.3f}"
		self.save_string   	=self.baseflow_string+'_m='+str(m)
		
		# Geometry
		self.mesh  = load_mesh(meshpath)
		self.BuildFunctionSpace()
		self.GenerateTestFunction()
		self.Trial = TrialFunction(self.Space)

		self.InitialConditions() # Main function space, start swirlless (initialises q)
		self.r = SpatialCoordinate(self.mesh)[1]
		# Physical parameters
		Re_s=.1
		# Sponged Reynolds number
		# Sponged Reynolds number
		self.Re=interpolate(Expression("x[0]<=70 && x[1]<=10 ? "+str(Re)+" : "+\
									   "x[0]<=70 && x[1]> 10 ? "+str(Re)+"+("+str(Re_s)+"-"+str(Re)+")*.5*(1+tanh(4*tan(-pi/2+pi*abs(x[1]-10)/50))) : "+\
									   "x[0]> 70 && x[1]<=10 ? "+str(Re)+"+("+str(Re_s)+"-"+str(Re)+")*.5*(1+tanh(4*tan(-pi/2+pi*abs(x[0]-70)/50))) : "+\
									   							 str(Re)+"+("+str(Re_s)+"-"+str(Re)+")*.5*(1+tanh(4*tan(-pi/2+pi*abs(x[1]-10)/50)))+"+\
																		  "("+str(Re_s)+"-"+str(Re)+"-("+str(Re_s)+"-"+str(Re)+")*.5*(1+tanh(4*tan(-pi/2+pi*abs(x[1]-10)/50))))*.5*(1+tanh(4*tan(-pi/2+pi*abs(x[0]-70)/50))) ",
									   degree=2), FunctionSpace(self.mesh,"Lagrange",1))

		# Memoisation routine - find closest in Re and S
		file_names = [f for f in os.listdir(self.dnspath+self.private_path) if f[:-3]=='xml']
		closest_file_name=self.dnspath+"last_baseflow.xml"
		d=1e12
		for file_name in file_names:
			for entry in file_name[:-4].split('_'):
				if entry[0]=='S': 	Sd =float(entry[2:])
				#if entry[:1]=='Re': Red=float(entry[3:])
			fd=abs(S-Sd)#+abs(Re-Red)
			if fd<d: d,closest_file_name=fd,self.dnspath+self.private_path+file_name

		File(closest_file_name) >> self.q.vector()

	def BuildFunctionSpace(self):
		# Taylor Hodd elements ; stable element pair
		FE_vector=VectorElement("Lagrange",self.mesh.ufl_cell(),2,3)
		FE_scalar=FiniteElement("Lagrange",self.mesh.ufl_cell(),1)
		self.Space=FunctionSpace(self.mesh,MixedElement([FE_vector,FE_scalar]))

	def GenerateTestFunction(self):
		testFunction = TestFunction(self.Space)
		self.Test = [as_vector((testFunction[0],testFunction[1],testFunction[2])),testFunction[3]]

	def InitialConditions(self): self.q = interpolate(Constant([1,0,0,0]), self.Space)

	def BoundaryConditions(self,S):
		symmetry,inlet,outlet,misc=boundary_geometry()
		# define boundary conditions for Newton/timestepper
		uth = Expression(str(S)+'*(x[1]<1 ? x[1]*(2-x[1]*x[1]) : 1/x[1])', degree=2)
		bcs_symmetry_r	= DirichletBC(self.Space.sub(0).sub(1), 0,   symmetry)  # Derivated from momentum eqs as r->0
		bcs_symmetry_th	= DirichletBC(self.Space.sub(0).sub(2), 0,   symmetry)
		bcs_misc_r  	= DirichletBC(self.Space.sub(0).sub(1), 0,   misc)
		bcs_misc_th	  	= DirichletBC(self.Space.sub(0).sub(2),	0,   misc)  	# Free slip at top (x is Neumann as right hand side of variational formulation)
		bcs_inflow_x  	= DirichletBC(self.Space.sub(0).sub(0), 1,   inlet) 	# Constant inflow
		bcs_inflow_r  	= DirichletBC(self.Space.sub(0).sub(1), 0,   inlet) 	# No radial flow
		bcs_inflow_th	= DirichletBC(self.Space.sub(0).sub(2), uth, inlet) 	# Little theta flow
		self.bc = [bcs_symmetry_r,bcs_symmetry_th,bcs_misc_r,bcs_misc_th,bcs_inflow_x,bcs_inflow_r,bcs_inflow_th]
	
	def BoundaryConditionsPerturbations(self):
		symmetry,inlet,outlet,misc=boundary_geometry()
		bcs_symmetry    = DirichletBC(self.Space.sub(0), 	   (0,0,0), symmetry) # Derivated from momentum eqs as r->0
		bcs_symmetry_r  = DirichletBC(self.Space.sub(0).sub(1), 0, 	    symmetry) # Weaker conditions is m==0
		bcs_symmetry_th = DirichletBC(self.Space.sub(0).sub(2), 0, 	    symmetry)
		bcs_misc_r   	= DirichletBC(self.Space.sub(0).sub(1), 0,    	misc)
		bcs_misc_th	 	= DirichletBC(self.Space.sub(0).sub(2), 0,   	misc)     # Free slip at top (x is Neumann as right hand side of variational formulation)
		bcs_inflow_r 	= DirichletBC(self.Space.sub(0).sub(1), 0,	    inlet)    # No radial flow
		self.bcp = [bcs_misc_r,bcs_misc_th,bcs_inflow_r]
		if self.m==0: self.bcp.extend([bcs_symmetry_r,bcs_symmetry_th])
		else:		  self.bcp.append(bcs_symmetry)

	def ComputeIndices(self):
		# Collect all dirichlet boundary dof indices
		bcinds = []
		for b in self.bcp:
			bcdict = b.get_boundary_values()
			bcinds.extend(bcdict.keys())

		# total number of dofs
		N = self.Space.dim()

		# indices of free nodes
		self.freeinds = np.setdiff1d(range(N),bcinds,assume_unique=True).astype(np.int32)

	def NonlinearOperator(self):
		u,p=extract_up(self.q)
		r,test_u,test_m=self.r,self.Test[0],self.Test[1]
		
		#mass (variational formulation)
		F = div_re(u,r)*test_m*r*dx
		#momentum (different test functions and IBP)
		F += 		   dot(grad_re(u,r)*u,  	 test_u)   *r*dx # Convection
		F += self.mu*inner(grad_re(u,r), grad_re(test_u,r))*r*dx # Diffusion
		F -= 		   dot(p, 			  div_re(test_u,r))*r*dx # Pressure
		return F

	def NonlinearOperatorComplex(self):
		u_r,p_r=extract_up(self.q_c)
		u_0,p_0=extract_up(self.q)
		m,r,test_u,test_m=self.m,self.r,self.Test[0],self.Test[1]
		
		#mass (variational formulation)
		F  = div_re(u_r,r)*test_m*r*dx
		#momentum (different test functions and IBP)
		F += 	 	   dot(grad_re(u_r,r)*u_0,     	 test_u) 	 *r*dx # Convection
		F +=     	   dot(grad_re(u_0,r)*u_r,     	 test_u) 	 *r*dx
		F += self.mu*inner(grad_re(u_r,r),   grad_re(test_u,r))  *r*dx # Diffusion
		F -= self.mu*inner(grad_im(u_r,r,m), grad_im(test_u,r,m))*r*dx
		F -= 		   dot(p_r, 			  div_re(test_u,r))  *r*dx # Pressure
		return F

	def JacobianNonlinearOperatorImaginary(self):
		u_r,p_r=extract_up(self.q_c)
		u_0,p_0=extract_up(self.q)
		m,r,test_u,test_m=self.m,self.r,self.Test[0],self.Test[1]
		
		#mass (imaginary part)
		F  = div_im(u_r,r,m)*test_m*r*dx
		#momentum (imaginary part)
		F +=	 	   dot(grad_im(u_r,r,m)*u_0,     test_u) 	 *r*dx # Convection
		F += 	 	   dot(grad_im(u_0,r,m)*u_r,     test_u) 	 *r*dx
		F += self.mu*inner(grad_im(u_r,r,m), grad_re(test_u,r))  *r*dx # Diffusion
		F += self.mu*inner(grad_re(u_r,r),   grad_im(test_u,r,m))*r*dx
		F -= 		   dot(p_r, 			  div_im(test_u,r,m))*r*dx
		return F

	def Newton(self):
		if self.n_S>1: Ss= np.cos(np.pi*np.linspace(self.n_S,0,self.n_S)/2/self.n_S)*self.S # Chebychev spacing
		else: 		   Ss=[self.S]
		for S_current in Ss: 	# Increase swirl
			for nu_current in np.linspace(self.nu,1,self.n_nu): # Decrease viscosity (non physical but helps CV)
				print("viscosity prefactor: ", nu_current)
				print("swirl intensity: ",	    S_current)
				self.mu=nu_current/self.Re #recalculate viscosity with prefactor
				self.BoundaryConditions(S_current) #for temporal-dependant boundary condition
				base_form  = self.NonlinearOperator() #no azimuthal decomposition for base flow (so no imaginary part to operator)
				dbase_form = derivative(base_form, self.q, self.Trial)
				solve(base_form == 0, self.q, self.bc, J=dbase_form, solver_parameters={"newton_solver":{'linear_solver' : 'mumps','relaxation_parameter':rp,"relative_tolerance":1e-12,'maximum_iterations':30,"absolute_tolerance":ae}})
				if nu_current==1:
					#write results in private_path for a given mu
					u_r,p_r = self.q.split()
					File(self.dnspath+self.private_path+"u_S="		 +f"{S_current:00.3f}"+".pvd") << u_r
					File(self.dnspath+self.private_path+"baseflow_S="+f"{S_current:00.3f}"+".xml") << self.q.vector()
					print(".xml written!")
				
		#write result of current mu
		File( self.dnspath+"last_u.pvd") << u_r
		File( self.dnspath+"last_baseflow.xml") << self.q.vector()
		print(self.dnspath+"last_baseflow.xml written!")

	def ComputeAM(self):
		#parameters['linear_algebra_backend'] = 'Eigen'

		# Go complex
		# Taylor Hodd elements ; stable element pair
		FE_vector=VectorElement("Lagrange",self.mesh.ufl_cell(),2,3)
		FE_scalar=FiniteElement("Lagrange",self.mesh.ufl_cell(),1)
		self.q_c = Coefficient(FunctionSpace(self.mesh,MixedElement([FE_vector,FE_scalar,FE_vector,FE_scalar])))

		#matrix A (m*m): Jacobian calculated by hand
		Aa = as_backend_type(assemble(self.JacobianNonlinearOperatorReal())).sparray()
		#Aa = sps.csr_matrix((data, indices, indptr))
		if self.m!=0:
			Aa_imag = as_backend_type(assemble(self.JacobianNonlinearOperatorImaginary())).sparray()
			Aa = Aa.view(dtype=np.complex)+1j*Aa_imag.view(dtype=np.complex)

		self.A = Aa[self.freeinds,:][:,self.freeinds].tocsc()

		#forcing norm M (m*m): here we choose ux^2+ur^2+uth^2 as forcing norm
		#other userdefined norm can be used, to be added later
		up_r = as_vector((self.Trial[0],self.Trial[1],self.Trial[2]))
		#up_i = as_vector((	 Trial_i[0],   Trial_i[1],   Trial_i[2]))
		M_form=dot(up_r,self.Test[0])*self.r*dx
		#M_form=dot(up_r,self.Test[0])*r*dx+dot(up_i,self.Test[0])*self.r*dx
		Ma = assemble(M_form)
		Ma = as_backend_type(Ma).sparray().tocsc()
		self.M = Ma[self.freeinds,:][:,self.freeinds]

	def Getw0(self):
		U,p=self.q.split(deepcopy=True)
		u,v,w=U.split(deepcopy=True)
		u=u.vector().get_local()
		return np.min(u)

	def Resolvent(self,k,freq_list):
		print("check base flow max and min in u:",np.max(self.q.vector()[:]),",",np.min(self.q.vector()[:]))

		#matrix B (m*m): with matrix A form altogether the resolvent operator
		up=as_vector((self.Trial[0],self.Trial[1],self.Trial[2]))
		pp=self.Trial[3]
		B_form=dot(up,self.Test[0])*self.r*dx
		Ba = assemble(B_form)
		Ba = as_backend_type(Ba).sparray().tocsc()

		#response norm Mr (m*m): here we choose the same as forcing norm
		Mr, Mf = self.M, self.M

		#quadrature Q (m*m): it is required to compensate the quadrature in resolvent operator R, because R=(A-i*omegaB)^(-1)
		Q_form=dot(up,self.Test[0])*self.r*dx+pp*self.Test[1]*self.r*dx
		Qa = assemble(Q_form)
		Qa = as_backend_type(Qa).sparray().tocsc()

		#matrix P (m*n) reshapes forcing vector (n*1) to (m*1). In principal, it contains only 0 and 1 elements.
		#It can also restrict the flow regions of forcing, to be implemented later. 
		#A note for very rare case: if one wants to damp but not entirely eliminate the forcing in some regions (not the case here), one can put values between 0 and 1. In that case, the matrix I in the following is not the same as P. 
		
		index_forcing=self.Space.sub(0).dofmap().dofs() #get all index related to u
		index_forcing.sort()

		row_ind=np.intersect1d(self.freeinds,index_forcing) #get free index related to u
		row_ind.sort()
		m=len(self.Space.dofmap().dofs())
		n=len(row_ind)
		col_ind=np.arange(n)
		Pa=sps.csc_matrix((np.ones(n),(row_ind,col_ind)),(m,n))

		#matrix I (m*n) reshapes forcing matrix Mf (m*m) to I^T*Mf*I (n*n). The matrix I can be different from P in that very rare case remarked above.
		Ia=Pa

		B  = Ba[self.freeinds,:][:,self.freeinds]
		P  = Pa[self.freeinds]
		I  = Ia[self.freeinds]
		Q  = Qa[self.freeinds,:][:,self.freeinds]

		Q_shape=np.shape(Q)
		print('matrix Q size: '+str(Q_shape))
		P_shape=np.shape(P)
		print('matrix P size: '+str(P_shape))

		for freq in freq_list:
			R = la.splu(-self.A-2*np.pi*1j*freq*B,permc_spec=3)
			# get response linear operator P^H*Q^H*R^H*Mr*R*Q*P
			def lhs(f):
				return P.transpose()*Q.transpose()*R.solve(Mr*R.solve(Q*P*f),trans='H')

			LHS = la.LinearOperator((min(P_shape),min(P_shape)),matvec=lhs,dtype='complex')

			# forcing linear operator is on the rhs M=I.transpose()*Mf*I
			gains,eigenvectors = la.eigs(LHS, k=k, M=I.transpose()*Mf*I, sigma=None,  maxiter=100, tol=ae, return_eigenvectors=True)
			
			#write forcing and response
			f=eigenvectors
			r=R.solve(Q*P*f)

			ua = Function(self.Space) #declaration for efficiency

			for i in range(k):
				ua.vector()[self.freeinds] = np.abs(P*f[:,i])
				u,p  = ua.split()
				File(self.dnspath+self.resolvent_path+"forcing_u"+self.save_string+"f="+f"{freq:00.3f}"+"_n="+f"{i+1:1d}"+".pvd") << u
				ua.vector()[self.freeinds] = np.abs(r[:,i])
				u,p  = ua.split()
				File(self.dnspath+self.resolvent_path+"response_u"+self.save_string+"f="+f"{freq:00.3f}"+"_n="+f"{i+1:1d}"+".pvd") << u
			
			#write gains
			np.savetxt(self.dnspath+self.resolvent_path+"gains"+self.save_string+"f="+f"{freq:00.3f}"+".dat",np.real(gains))

	def Eigenvalues(self,sigma,k,flag_mode,savematt,loadmatt):
		print("check base flow max and min in u:",np.max(self.q.vector()[:]),",",np.min(self.q.vector()[:]))

		#RHS
		if flag_mode==0:
			print("save matrix to file "+savematt+self.save_string+".mat and quit!")
			from scipy.io import savemat
			mdic = {"A": self.A, "M": self.M}
			savemat(savematt+self.save_string+".mat", mdic)
			return 0
		elif flag_mode==1:
			print("load matlab result from file "+loadmatt)
			from scipy.io import loadmat
			mdic=loadmat(loadmatt)
			vecs=mdic['V'] #if KeyError: 'V', it means the eigenvalue results are not saved into .mat
			vals=np.diag(mdic['D'])
		elif flag_mode==2:			
			print("Computing eigenvalues/vectors in Python!")
			ncv = max(10,2*k)
			vals, vecs = la.eigs(self.A, k=k, M=self.M, sigma=sigma, maxiter=60, tol=ae, ncv=ncv)
		else:
			print("Operation mode for eigenvalues is not correct. Nothing done.")
			return 0

		# only writing real parts of eigenvectors to file
		ua = Function(self.Space)
		flag_video=0 #1: export animation
		for i in range(0,k+1,k//10+1):
			ua.vector()[self.freeinds] = vecs[:,i].real

			u,p  = ua.split()
			File(self.dnspath+self.eig_path+"evec_u"  +self.save_string+"_n="+str(i+1)+".pvd") << u
			File(self.dnspath+self.eig_path+"evec_p"  +self.save_string+"_n="+str(i+1)+".pvd") << p
			if flag_video: # export animation
				print("Exporting video for eig "+str(i+1))
				angSteps = 20
				angList = list(2*np.pi/angSteps*np.arange(0,angSteps+1))

				angle0=np.angle(vecs[:,i])
				abs0=np.absolute(vecs[:,i])
				for k in range(0,angSteps+1):
					angle = angList[k]-angle0
					amp = abs0*np.cos(angle)
					ua.vector()[self.freeinds] = amp
					if self.label=='lowMach':
						u,p,rho  = ua.split()
					if self.label=='lowMach_reacting':
						u,p,rho,y  = ua.split()
					File(self.dnspath+self.eig_path+"anim_rho_nu="+f"{self.nu:00.3f}"+"_S="+f"{self.S:00.3f}"+"_"+str(i+1)+"_"+str(k)+".pvd") << rho
					File(self.dnspath+self.eig_path+"anim_u_nu="+f"{self.nu:00.3f}"+"_S="+f"{self.S:00.3f}"+"_"+str(i+1)+"_"+str(k)+".pvd") << u
					if self.label=='lowMach_reacting':
						File(self.dnspath+self.eig_path+"anim_y_nu="+f"{self.nu:00.3f}"+"_S="+f"{self.S:00.3f}"+"_"+str(i+1)+"_"+str(k)+".pvd") << y
		#write eigenvalues
		np.savetxt(self.dnspath+self.eig_path+"evals"+self.save_string+".dat",np.column_stack([vals.real, vals.imag]))