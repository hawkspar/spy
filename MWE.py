import dolfinx as dfx, ufl
import numpy as np
from slepc4py import SLEPc as slp
from mpi4py.MPI import COMM_WORLD

n=100
mesh=dfx.UnitSquareMesh(COMM_WORLD,n,n)
FE=ufl.FiniteElement("Lagrange",mesh.ufl_cell(),1)
V=dfx.FunctionSpace(mesh,FE)
u=ufl.TrialFunction(V)
v=ufl.TestFunction(V)

# Defining the forms
a=ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx
b=ufl.inner(         u,          v )*ufl.dx

zero=dfx.Function(V)
with zero.vector.localForm() as zero_loc: zero_loc.set(0)
# Identify BC DoFs
dim = mesh.topology.dim - 1
mesh.topology.create_connectivity(dim, mesh.topology.dim)
boundary = np.where(np.array(dfx.cpp.mesh.compute_boundary_facets(mesh.topology)) == 1)[0]
dofs = dfx.fem.locate_dofs_topological(V, dim, boundary)
bcs = dfx.DirichletBC(zero, dofs) # u_i=value at boundary

# Matrices
A = dfx.fem.assemble_matrix(a,bcs=[bcs]); A.assemble()
print(dofs)
print(A.getValues(range(5,11),range(5,11)))

# Same for B
B = dfx.fem.assemble_matrix(b,bcs=[bcs],diagonal=0); B.assemble()

# Solver
E = slp.EPS(); E.create()
E.setOperators(A,B) # Solve Ax=sigma*Mx
E.setWhichEigenpairs(E.Which.TARGET_MAGNITUDE) # Find eigenvalues close to sigma
E.setTarget(2*np.pi**2)
E.setDimensions(5,10) # Find k eigenvalues only with max number of Lanczos vectors
E.setTolerances(1e-3,60) # Set absolute tolerance and number of iterations
E.setProblemType(slp.EPS.ProblemType.GHEP)
# Spectral transform
ST = E.getST()
ST.setType(ST.Type.SINVERT)
# Krylov subspace
KSP = ST.getKSP()
KSP.setType('preonly')
# Preconditioner
PC =  KSP.getPC()
PC.setType('lu')
PC.setFactorSolverType('mumps')
#PC.setFactorShift("nonzero",ae)
E.setFromOptions()
E.solve()
n=E.getConverged()

# Solver
ths=np.pi**2*np.array([2,5,5,8,10])

print("Theoretical value | Calculated one | Difference")
for i in range(n):
    exp=E.getEigenvalue(i)
    print(f"{ths[i]:.3f}"+" | "+f"{np.real(exp):.3f}"+" | "+f"{np.abs(ths[i]-exp):00.2e}")

