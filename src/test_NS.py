import ufl
import numpy as np
from petsc4py import PETSc as pet
from mpi4py.MPI import COMM_WORLD as comm

from dolfinx.io import XDMFFile
from dolfinx.nls.petsc import NewtonSolver
from dolfinx.mesh import create_unit_square
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.fem import Function, FunctionSpace, dirichletbc, locate_dofs_geometrical

domain = create_unit_square(comm, 100, 100)
p0 = comm.rank == 0

rtol,atol=1e-6,1e-9

# -------------------------------------------------------------------------------------------
# Domain and Functions
# -------------------------------------------------------------------------------------------

r = ufl.SpatialCoordinate(domain)[1]
R=.3

def axis( x) -> np.ndarray: return np.isclose(x[1],0,atol) # Axis of symmetry at r=0
def inlet(x) -> np.ndarray: return np.isclose(x[0],0,atol) # Left border

TH0 = ufl.VectorElement("CG",domain.ufl_cell(),2,3)
TH1 = ufl.FiniteElement("CG",domain.ufl_cell(),1)
TH  = FunctionSpace(domain,TH0*TH1)

Q,T=Function(TH),ufl.TestFunction(TH)
U,P=ufl.split(Q)
v,s=ufl.split(T)

# -------------------------------------------------------------------------------------------
# Main form
# -------------------------------------------------------------------------------------------

# Operators with r multiplication
def grd(v,i:int=0):
	return ufl.as_tensor([[v[0].dx(0), i*v[0]/r+v[0].dx(1),  0],
						  [v[1].dx(0), i*v[1]/r+v[1].dx(1), -v[2]/r],
						  [v[2].dx(0), i*v[2]/r+v[2].dx(1),  v[1]/r]])

def div(v,i:int=0): return v[0].dx(0) + (1+i)*v[1]/r + v[1].dx(1)/r

# Navier Stokes equations multiplied by r^2
NS =ufl.inner(div(U),             s)
NS+=ufl.inner(grd(U)*U,           v)
NS-=ufl.inner(   P,           div(v))
NS+=ufl.inner(grd(U)+grd(U).T,grd(v))/100
NS*=r
NS*=ufl.dx

# -------------------------------------------------------------------------------------------
# Boundary conditions
# -------------------------------------------------------------------------------------------

# Handler
def constantBC(direction:int, boundary:bool) -> tuple:
    subspace=TH.sub(0).sub(direction)
    subspace_collapsed,_=subspace.collapse()
    # Compute unflattened DoFs (don't care for flattened ones)
    dofs = locate_dofs_geometrical((subspace, subspace_collapsed), boundary)
    cst = Function(subspace_collapsed)
    cst.interpolate(lambda x: np.zeros_like(x[0]))
    # Actual BCs
    return dirichletbc(cst, dofs, subspace) # u_i=value at boundary

# Compute DoFs
sub_space_x=TH.sub(0).sub(0)
sub_space_x_collapsed,_=sub_space_x.collapse()

u_inlet_x=Function(sub_space_x_collapsed)
u_inlet_x.interpolate(lambda x: np.tanh(6*(1-(x[1]/R)**2))*(x[1]<R)+
                            .05*np.tanh(6*((x[1]/R)**2-1))*(x[1]>R))

# Degrees of freedom
dofs_inlet_x = locate_dofs_geometrical((sub_space_x, sub_space_x_collapsed), inlet)
bcs_inlet_x = dirichletbc(u_inlet_x, dofs_inlet_x, sub_space_x) # Same as OpenFOAM

bcs=[bcs_inlet_x,constantBC(1,inlet),constantBC(2,inlet),constantBC(1,axis),constantBC(2,axis)]
#bcs=[bcs_inlet_x,constantBC(2,inlet),constantBC(1,axis),constantBC(2,axis)]

problem = NonlinearProblem(NS, Q, bcs=bcs)
solver  = NewtonSolver(comm, problem)

# -------------------------------------------------------------------------------------------
# Nonlinear solver
# -------------------------------------------------------------------------------------------

# Fine tuning
solver.convergence_criterion = "incremental"
solver.relaxation_parameter=.95 # Absolutely crucial for convergence
solver.max_iter=100
solver.rtol,solver.atol=rtol,atol
ksp = solver.krylov_solver
opts = pet.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "preonly"
opts[f"{option_prefix}pc_type"] = "lu"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()
if p0: print("Solver launch...",flush=True)
# Actual heavyweight
solver.solve(Q)

# Print output
U,_=Q.split()

with XDMFFile(comm, "test_NS.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(U)