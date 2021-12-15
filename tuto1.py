import numpy
import ufl
from mpi4py import MPI
from dolfinx.generation import UnitSquareMesh
mesh = UnitSquareMesh(MPI.COMM_WORLD, 8, 8)
from dolfinx.fem import FunctionSpace
V = FunctionSpace(mesh, ("CG", 1))
from dolfinx.fem import Function
uD = Function(V)
uD.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
fdim = mesh.topology.dim - 1
# Create facet to cell connectivity required to determine boundary facets
from dolfinx.cpp.mesh import compute_boundary_facets
mesh.topology.create_connectivity(fdim, mesh.topology.dim)
boundary_facets = numpy.where(numpy.array(compute_boundary_facets(mesh.topology)) == 1)[0]
from dolfinx.fem import locate_dofs_topological, DirichletBC
boundary_dofs = locate_dofs_topological(V, fdim, boundary_facets)
bc = DirichletBC(uD, boundary_dofs)
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
from dolfinx.fem import Constant
f = Constant(mesh, -6)
a = ufl.dot(ufl.grad(u), ufl.grad(v)) * ufl.dx
L = f * v * ufl.dx
from dolfinx.fem import LinearProblem
problem = LinearProblem(a, L, bcs=[bc], petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
uh = problem.solve()
V2 = FunctionSpace(mesh, ("CG", 2))
uex = Function(V2)
uex.interpolate(lambda x: 1 + x[0]**2 + 2 * x[1]**2)
from dolfinx.fem import assemble_scalar
L2_error = ufl.inner(uh - uex, uh - uex) * ufl.dx
error_L2 = numpy.sqrt(assemble_scalar(L2_error))
u_vertex_values = uh.compute_point_values()
u_ex_vertex_values = uex.compute_point_values()
error_max = numpy.max(numpy.abs(u_vertex_values - u_ex_vertex_values))
print(f"Error_L2 : {error_L2:.2e}")
print(f"Error_max : {error_max:.2e}")