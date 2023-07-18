import os, ufl, re
import numpy as np
import dolfinx as dfx
from dolfinx.fem import Function
from petsc4py import PETSc as pet
from slepc4py import SLEPc as slp
from mpi4py.MPI import COMM_WORLD as comm

p0=comm.rank==0

# Cylindrical operators
def grd(r,dx:int,dr:int,dt:int,v,m:int):
	if len(v.ufl_shape)==0: return ufl.as_vector([v.dx(dx), v.dx(dr), m*1j*v/r])
	return ufl.as_tensor([[v[dx].dx(dx), v[dx].dx(dr),  m*1j*v[dx]		 /r],
						  [v[dr].dx(dx), v[dr].dx(dr), (m*1j*v[dr]-v[dt])/r],
						  [v[dt].dx(dx), v[dt].dx(dr), (m*1j*v[dt]+v[dr])/r]])

def div(r,dx:int,dr:int,dt:int,v,m:int):
	if len(v.ufl_shape)==1: return v[dx].dx(dx) + (r*v[dr]).dx(dr)/r + m*1j*v[dt]/r
	return ufl.as_vector([v[dx,dx].dx(dx)+v[dr,dx].dx(dr)+v[dr,dx]+ m*1j*v[dt,dx]		   /r,
						  v[dx,dr].dx(dx)+v[dr,dr].dx(dr)+v[dr,dr]+(m*1j*v[dt,dr]-v[dt,dt])/r,
						  v[dx,dt].dx(dx)+v[dr,dt].dx(dr)+v[dr,dt]+(m*1j*v[dt,dt]+v[dt,dr])/r])

def crl(r,dx:int,dr:int,dt:int,mesh:ufl.Mesh,v,m:int,i:int=0):
	return ufl.as_vector([(i+1)*v[dt]		+r*v[dt].dx(dr)-m*dfx.fem.Constant(mesh, 1j)*v[dr],
    m*dfx.fem.Constant(mesh,1j)*v[dx]		-  v[dt].dx(dx),
								v[dr].dx(dx)-i*v[dx]-v[dx].dx(dr)])

# Helpers
def dirCreator(path:str) -> None:
	if not os.path.isdir(path):
		if p0: os.mkdir(path)
	comm.barrier() # Wait for all other processors

def checkComm(f:str) -> bool:
	match = re.search(r'n=(\d*)',f)
	if int(match.group(1))!=comm.size: return False
	match = re.search(r'p=([0-9]*)',f)
	if int(match.group(1))!=comm.rank: return False
	return True

# Simple handler
def meshConvert(path:str,cell_type:str='triangle',prune=True) -> None:
	import meshio #pip3 install h5py meshio
	gmsh_mesh = meshio.read(path+".msh")
	# Write it out again
	ps = gmsh_mesh.points[:,:(3-prune)]
	cs = gmsh_mesh.get_cells_type(cell_type)
	dolfinx_mesh = meshio.Mesh(points=ps, cells={cell_type: cs})
	meshio.write(path+".xdmf", dolfinx_mesh)
	print("Mesh "+path+".msh converted to "+path+".xdmf !",flush=True)

# Naive save with dir creation
def saveStuff(dir:str,name:str,fun:Function) -> None:
	dirCreator(dir)
	proc_name=dir+name.replace('.',',')+f"_n={comm.size:d}_p={comm.rank:d}"
	fun.x.scatter_forward()
	np.save(proc_name,fun.x.array)
	if p0: print("Saved "+proc_name+".npy",flush=True)

# Memoisation routine - find closest in param
def findStuff(path:str,params:dict,format=lambda f:True,distributed=True):
	closest_file_name=path
	d=np.infty
	for file_name in os.listdir(path):
		if format(file_name) and (not distributed or checkComm(file_name)): # Lazy evaluation !
			fd=0 # Compute distance according to all params
			for param in params:
				regexp='\d*,?\d*e?\+?-?\d*'
				match = re.search(param+r'=('+regexp+'\+?-?'+regexp+'j?)',file_name) # Can handle everything from negative integers to complex numbers in scientific notation
				param_file = complex(match.group(1).replace(',','.')) # Take advantage of file format
				fd += abs(params[param]-param_file)
			if fd<d: d,closest_file_name=fd,path+file_name
	return closest_file_name

def loadStuff(path:str,params:dict,fun:Function) -> None:
	closest_file_name=findStuff(path,params,lambda f: f[-3:]=="npy")
	if p0: print("Loading "+closest_file_name,flush=True)
	fun.x.array[:]=np.load(closest_file_name,allow_pickle=True)
	fun.x.scatter_forward()

# Wrapper
def assembleForm(form:ufl.Form,bcs:list=[],sym=False,diag=0) -> pet.Mat:
	# JIT options for speed
	form = dfx.fem.form(form, jit_options={"cffi_extra_compile_args": ["-Ofast", "-march=native"], "cffi_libraries": ["m"]})
	M = dfx.cpp.fem.petsc.create_matrix(form)
	M.setOption(M.Option.IGNORE_ZERO_ENTRIES, 1)
	M.setOption(M.Option.SYMMETRY_ETERNAL, sym)
	dfx.fem.petsc._assemble_matrix_mat(M, form, bcs, diag)
	M.assemble()
	return M

# PETSc Matrix free method
def pythonMatrix(dims:list,py,comm) -> pet.Mat:
	M = pet.Mat().create(comm)
	M.setSizes(dims)
	M.setType(pet.Mat.Type.PYTHON)
	M.setPythonContext(py)
	M.setUp()
	return M

# Krylov subspace
def configureKSP(KSP:pet.KSP,params:dict,icntl:bool=False) -> None:
	KSP.setTolerances(rtol=params['rtol'], atol=params['atol'], max_it=params['max_iter'])
	# Direct solver
	KSP.setType('preonly')
	# Brutal LU preconditioner (performance nightmare !)
	PC = KSP.getPC(); PC.setType('lu')
	PC.setFactorSolverType('mumps')
	KSP.setFromOptions()
	if icntl: PC.getFactorMatrix().setMumpsIcntl(14,1000)

# Eigenvalue problem solver
def configureEPS(EPS:slp.EPS,k:int,params:dict,pb_type:slp.EPS.ProblemType,shift:bool=False,app=None) -> None:
	EPS.setDimensions(k,max(10,2*k)) # Find k eigenvalues only with max number of Lanczos vectors
	EPS.setTolerances(params['atol'],params['max_iter']) # Set absolute tolerance and number of iterations
	EPS.setProblemType(pb_type)
	# Spectral transform
	ST = EPS.getST()
	if shift:
		ST.setType('sinvert')
		ST.getOperator() # CRITICAL TO MUMPS ICNTL
		configureKSP(ST.getKSP(),params,shift)
	else:
		KSP = ST.getKSP()
		KSP.setTolerances(rtol=params['rtol'], atol=params['atol'], max_it=params['max_iter'])
		KSP.setType('cg')
		ST.setPreconditionerMat(app)
		PC = KSP.getPC(); PC.setType('lu')
		PC.setFactorSolverType('mumps')
		KSP.setFromOptions()
	EPS.setFromOptions()

def azimuthalExtension(th,m,F,G=None,H=None,real=True,outer=True,cartesian=False):
	if outer: F  = np.outer(F,np.exp(1j*m*th)) # Operating on a cut plane
	else: 	  F *= 			  np.exp(1j*m*th)	 # Already in 3D
	if G is None:
		if real: return F.real.astype(np.float64)
		else: 	 return F.astype(np.complex64)
	if outer:
		G = np.outer(G,np.exp(1j*m*th))
		H = np.outer(H,np.exp(1j*m*th))
		th = np.tile(th,(G.shape[0],1))
	else:
		G *= np.exp(1j*m*th)
		H *= np.exp(1j*m*th)
	if cartesian:
		G,H = np.cos(th)*G-np.sin(th)*H,np.sin(th)*G+np.cos(th)*H # Moving to Cartesian referance frame at last moment
	if real: return [I.real.astype(np.float64) for I in [F,G,H]] # Reduce memory footprint
	else: 	 return [I.astype(np.complex64)    for I in [F,G,H]]