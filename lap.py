import dolfinx, ufl
import numpy as np
import scipy.sparse as sps
import scipy.sparse.linalg as la
from mpi4py.MPI import COMM_WORLD

n=100
mesh=dolfinx.UnitSquareMesh(COMM_WORLD,n,n)
FE=ufl.FiniteElement("Lagrange",mesh.ufl_cell(),1)
V=dolfinx.FunctionSpace(mesh,FE)
u=ufl.TrialFunction(V)
v=ufl.TestFunction(V)

# Defining the forms
a=ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx
b=ufl.inner(         u,          v )*ufl.dx

# Identify BC DoFs
dim = mesh.topology.dim - 1
mesh.topology.create_connectivity(dim, mesh.topology.dim)
boundary = np.where(np.array(dolfinx.cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]
boundary_dofs = dolfinx.fem.locate_dofs_topological(V, dim, boundary)
# Total number of DoFs
N = V.dofmap.index_map.size_global * V.dofmap.index_map_bs
# Free DoFs
freeinds = np.setdiff1d(range(N),boundary_dofs,assume_unique=True).astype(np.int32)

# Matrices
A = dolfinx.fem.assemble_matrix(a); A.assemble()
# Convert from PeTSc to Scipy for easy slicing
ai, aj, av = A.getValuesCSR()
# Slice away BCs
A = sps.csr_matrix((av, aj, ai))[freeinds,:][:,freeinds]

# Same for B
B = dolfinx.fem.assemble_matrix(b); B.assemble()
bi, bj, bv = B.getValuesCSR()
B = sps.csr_matrix((bv, bj, bi))[freeinds,:][:,freeinds]

# Solver
k=5
vals, vecs = la.eigs(A, k=k, M=B, sigma=0)
ths=np.pi**2*np.array([2,5,5,8,10])

print("Theoretical value | Calculated one | Difference")
for i in range(k): print(f"{ths[i]:1.14f}"+" | "+f"{np.real(vals[i]):.4f}"+f"{np.imag(vals[i]):+.0e}"+"j | "+f"{np.abs(ths[i]-vals[i]):00.2e}")