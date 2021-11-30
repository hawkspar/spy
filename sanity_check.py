import numpy as np
from pdb import set_trace
import dolfinx as dfx, ufl
import scipy.sparse as sps
from dolfinx.io import VTKFile
import scipy.sparse.linalg as la
from mpi4py.MPI import COMM_WORLD

n=3
mesh=dfx.UnitSquareMesh(COMM_WORLD,n,n)
FE=ufl.VectorElement("Lagrange",mesh.ufl_cell(),1)
V=dfx.FunctionSpace(mesh,FE)
u=ufl.TrialFunction(V)
u0,u1=ufl.split(u)
v=ufl.TestFunction(V)
v0,v1=ufl.split(v)

# Defining the forms
a=ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx
b=ufl.inner(         u0,         v0)*ufl.dx

def bottom(x): return np.isclose(x[1],0)
def right(x):  return np.isclose(x[0],1)
def top(x):    return np.isclose(x[1],1)
def left(x):   return np.isclose(x[0],0)
boundaries=[bottom,right,top,left]

def ConstantBC(direction, boundary) -> tuple:
    sub_space=V.sub(direction)
    sub_space_collapsed=sub_space.collapse()
    # Compute proper zeros
    constant=dfx.Function(sub_space_collapsed)
    with constant.vector.localForm() as zero_loc: zero_loc.set(0)
    # Compute DoFs
    dofs = dfx.fem.locate_dofs_geometrical((sub_space, sub_space_collapsed), boundary)
    return dofs[0] # Only return unflattened dofs

# Identify BC DoFs
dofs=np.empty(0,dtype=np.int64)
for i in range(2):
    for bd in boundaries:
        dofs=np.union1d(dofs,ConstantBC(i,bd)).astype(np.int64)

# Sparse utilities
def csr_zero_rows(csr : sps.csr_matrix, rows : np.ndarray):
	for row in rows: csr.data[csr.indptr[int(row)]:csr.indptr[int(row)+1]] = 0

def csc_zero_cols(csc : sps.csc_matrix, cols : np.ndarray):
	for col in cols: csc.data[csc.indptr[int(col)]:csc.indptr[int(col)+1]] = 0

# Matrices
A = dfx.fem.assemble_matrix(a); A.assemble()
# Convert from PeTSc to Scipy for easy slicing
ai, aj, av = A.getValuesCSR()

A = sps.csr_matrix((av, aj, ai))
# Efficiently cancel out rows and columns
csr_zero_rows(A,dofs)
A=A.tocsc()
csc_zero_cols(A,dofs)
# Introduce a -1 to force homogeneous BC
A[dofs,dofs]=-1
A.eliminate_zeros()

# Same for B
B = dfx.fem.assemble_matrix(b); B.assemble()
bi, bj, bv = B.getValuesCSR()
B = sps.csr_matrix((bv, bj, bi))
# Efficiently cancel out rows and columns
csr_zero_rows(B,dofs)
B=B.tocsc()
csc_zero_cols(B,dofs)
B.eliminate_zeros()

set_trace()

# Solver
k=5
vals, vecs = la.eigs(-A, k=k, M=B, sigma=0)
# Write eigenvectors back in pvd
for i in range(vals.size):
    q=dfx.Function(V)
    q.vector.array = vecs[:,i]
    with VTKFile(COMM_WORLD, "u.pvd","w") as vtk:
        vtk.write([q._cpp_object])
"""
ths=np.pi**2*np.array([2,5,5,8,10])

print("Theoretical value | Calculated one | Difference")
for i in range(k): print(f"{ths[i]:1.14f}"+" | "+f"{np.real(vals[i]):.4f}"+f"{np.imag(vals[i]):+.0e}"+"j | "+f"{np.abs(ths[i]-vals[i]):00.2e}")
"""