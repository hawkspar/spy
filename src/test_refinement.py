import ufl
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc as pet
from dolfinx import mesh, io, fem, nls, log, geometry

def q(u): return 1 + u**2

domain = mesh.create_unit_square(MPI.COMM_WORLD, 10, 10)
p0 = domain.comm.rank == 0
x = ufl.SpatialCoordinate(domain)
u_ufl = 1 + x[0] + 2*x[1]
f = - ufl.div(q(u_ufl)*ufl.grad(u_ufl))

V = fem.FunctionSpace(domain, ("CG", 1))
u_exact = lambda x: eval(str(u_ufl))
u_D = fem.Function(V)
u_D.interpolate(u_exact)
fdim = domain.topology.dim - 1
boundary_facets = mesh.locate_entities_boundary(domain, fdim, lambda x: np.full(x.shape[1], True, dtype=bool))
bc = fem.dirichletbc(u_D, fem.locate_dofs_topological(V, fdim, boundary_facets))

uh = fem.Function(V)
v = ufl.TestFunction(V)
F = q(uh)*ufl.dot(ufl.grad(uh), ufl.grad(v))*ufl.dx - f*v*ufl.dx

problem = fem.petsc.NonlinearProblem(F, uh, bcs=[bc])

solver = nls.petsc.NewtonSolver(MPI.COMM_WORLD, problem)
solver.convergence_criterion = "incremental"
solver.rtol = 1e-6
solver.report = True

ksp = solver.krylov_solver
opts = pet.Options()
option_prefix = ksp.getOptionsPrefix()
opts[f"{option_prefix}ksp_type"] = "cg"
opts[f"{option_prefix}pc_type"] = "gamg"
opts[f"{option_prefix}pc_factor_mat_solver_type"] = "mumps"
ksp.setFromOptions()

#if p0: log.set_log_level(log.LogLevel.INFO)
n, converged = solver.solve(uh)
assert(converged)
if p0:print(f"Number of interations: {n:d}")

# Compute L2 error and error at nodes
V_ex = fem.FunctionSpace(domain, ("CG", 2))
u_ex = fem.Function(V_ex)
u_ex.interpolate(u_exact)
error_local = fem.assemble_scalar(fem.form((uh - u_ex)**2 * ufl.dx))
error_L2 = np.sqrt(domain.comm.allreduce(error_local, op=MPI.SUM))
if p0: print(f"L2-error: {error_L2:.2e}")

# Compute values at mesh vertices
error_max = domain.comm.allreduce(np.max(np.abs(uh.x.array -u_D.x.array)), op=MPI.MAX)
if p0: print(f"Error_max: {error_max:.2e}")

with io.XDMFFile(MPI.COMM_WORLD, "test_coarse.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

def high_error(x):
    x = x.T
    # Find cells whose bounding-box collide with the the points
    bbtree = geometry.BoundingBoxTree(domain, 2)
    cell_candidates = geometry.compute_collisions(bbtree, x)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, x)
    cells, points_on_proc = [], []
    for i, xp in enumerate(x):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(xp)
            cells.append(colliding_cells.links(i)[0])
    uh_points   = uh.eval(  points_on_proc, cells)
    u_ex_points = u_ex.eval(points_on_proc, cells)
    res_points = (uh_points-u_ex_points)**2
    max_res = domain.comm.allreduce(np.max(res_points), op=MPI.MAX)
    return res_points>.8*max_res

edges = mesh.locate_entities(domain, domain.topology.dim-1, high_error)
domain.topology.create_entities(1)
# Mesh refinement
domain = mesh.refine(domain, edges, redistribute=False)

def uhf(x):
    x = x.T
    # Find cells whose bounding-box collide with the the points
    bbtree = geometry.BoundingBoxTree(domain, 2)
    cell_candidates = geometry.compute_collisions(bbtree, x)
    # Choose one of the cells that contains the point
    colliding_cells = geometry.compute_colliding_cells(domain, cell_candidates, x)
    cells, points_on_proc = [], []
    for i, xp in enumerate(x):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(xp)
            cells.append(colliding_cells.links(i)[0])
    print('bip')
    return uh.eval(points_on_proc, cells)

V = fem.FunctionSpace(domain, ("CG", 1))
uh2 = fem.Function(V)
uh2.interpolate(uh)

with io.XDMFFile(MPI.COMM_WORLD, "test_fine.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)