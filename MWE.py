import dolfinx
from petsc4py import PETSc
import numpy as np
from mpi4py.MPI import COMM_WORLD
import ufl

mesh=dolfinx.UnitSquareMesh(COMM_WORLD,10,10,dolfinx.cpp.mesh.CellType.triangle)
r_max=.1
l=.01

FE_vector=ufl.VectorElement("Lagrange",mesh.ufl_cell(),2,3)
FE_scalar=ufl.FiniteElement("Lagrange",mesh.ufl_cell(),1)
Space=dolfinx.FunctionSpace(mesh,ufl.MixedElement([FE_vector,FE_scalar]))	# full vector function space
# Subspaces
U_space=Space.sub(0).collapse()
S = dolfinx.Constant(mesh,0)

# Modified vortex that goes to zero at top boundary
def grabovski_berger(r):
    psi=(r_max+l-r)/l/r_max
    mr=r<1
    psi[mr]=r[mr]*(2-r[mr]**2)
    ir=np.logical_and(r>=1,r<r_max)
    psi[ir]=1/r[ir]
    return psi
# 3D inlet flow
def inlet_flow(x):
    n=x[0].size
    return np.array([np.ones(n),np.zeros(n),S*grabovski_berger(x[1])],dtype=PETSc.ScalarType) # Swirl intensity determined at a higher level
u_inlet=dolfinx.Function(U_space)