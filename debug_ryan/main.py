import dolfinx # FEM in python
import matplotlib.pyplot as plt
import ufl # variational formulations
import numpy as np
from mpi4py import MPI
from petsc4py.PETSc import ScalarType
import gmsh # Mesh generation
import extract
from os.path import isfile

# Geometry
R_i = 1.0 # Radius of the inclusion
R_e = 6.9  # Radius of the matrix (whole domain)
aspect_ratio = 1.0 # start with a circle, otherwise ellipse

# Material
E_m = 0.8 # Young's modulus in matrix
nu_m = 0.35 # Poisson's ratio in matrix
E_i = 11.0 # Young's modulus of inclusion
nu_i = 0.3 # Poisson's ratio in inclusion

def solve_eshelby (N=2, degree=1, R_i=1, R_e=6.9, aspect_ratio=1, E_m=0.8, nu_m=0.35, E_i=11, nu_i=0.3):
	mesh_size = R_i/N
	mesh_order = 2

	mesh_comm = MPI.COMM_WORLD
	model_rank = 0
	
	if not isfile(f"/home/shared/debug_ryan/mesh/{N}.xdmf"):
		gmsh.initialize()
		facet_names = {"inner_boundary": 1, "outer_boundary": 2}
		cell_names = {"inclusion": 1, "matrix": 2}
		model = gmsh.model()
		model.add("Disk")
		model.setCurrent("Disk")
		gdim = 2 # geometric dimension of the mesh
		inner_disk = gmsh.model.occ.addDisk(0, 0, 0, R_i, aspect_ratio * R_i)
		outer_disk = gmsh.model.occ.addDisk(0, 0, 0, R_e, R_e)
		whole_domain = gmsh.model.occ.fragment([(gdim, outer_disk)], [(gdim, inner_disk)])
		gmsh.model.occ.synchronize()

		# Add physical tag for bulk
		inner_domain = whole_domain[0][0]
		outer_domain = whole_domain[0][1]
		model.addPhysicalGroup(gdim, [inner_domain[1]], tag=cell_names["inclusion"])
		model.setPhysicalName(gdim, inner_domain[1], "Inclusion")
		model.addPhysicalGroup(gdim, [outer_domain[1]], tag=cell_names["matrix"])
		model.setPhysicalName(gdim, outer_domain[1], "Matrix")

		# Add physical tag for boundaries
		lines = gmsh.model.getEntities(dim=1)
		inner_boundary = lines[1][1]
		outer_boundary = lines[0][1]
		gmsh.model.addPhysicalGroup(1, [inner_boundary], facet_names["inner_boundary"])
		gmsh.model.addPhysicalGroup(1, [outer_boundary], facet_names["outer_boundary"])
		gmsh.option.setNumber("Mesh.CharacteristicLengthMin",mesh_size)
		gmsh.option.setNumber("Mesh.CharacteristicLengthMax",mesh_size)
		model.mesh.generate(gdim)
		gmsh.option.setNumber("General.Terminal", 1)
		model.mesh.setOrder(mesh_order)
		gmsh.option.setNumber("General.Terminal", 0)

		# Import the mesh in dolfinx
		from dolfinx.io import gmshio
		domain, cell_tags, facet_tags = gmshio.model_to_mesh(model, mesh_comm, model_rank, gdim=gdim)
		domain.name = "composite"
		cell_tags.name = f"{domain.name}_cells"
		facet_tags.name = f"{domain.name}_facets"
		gmsh.finalize()
		
		with dolfinx.io.XDMFFile(mesh_comm, f"/home/shared/debug_ryan/mesh/{N}.xdmf", "w") as xdmf:
			xdmf.write_mesh(domain)
	else:
		with dolfinx.io.XDMFFile(mesh_comm, f"/home/shared/debug_ryan/mesh/{N}.xdmf", "w") as xdmf:
			domain = xdmf.read_mesh(name="composite")

	dx = ufl.Measure("dx", subdomain_data=cell_tags, domain=domain)
	one = dolfinx.fem.Constant(domain,ScalarType(1.))

	area_inclusion = dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * dx(1)))
	area_matrix = dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * dx(2)))
	area_domain = dolfinx.fem.assemble_scalar(dolfinx.fem.form(one * ufl.dx))
	area_inclusion, area_matrix, area_domain

	V = dolfinx.fem.VectorFunctionSpace(domain,("Lagrange", degree),dim=2)

	def eps(u): return ufl.sym(ufl.grad(u))

	I2 = ufl.Identity(2)

	# Hooke's law is written as the top of this notebook
	def sigma(eps, E, nu):
		mu = E/(2*(1+nu))
		lamb = 2*mu*nu/(1-2*nu)
		return lamb*ufl.tr(eps)*I2 + 2*mu*eps

	u = ufl.TrialFunction(V)
	u_bar = ufl.TestFunction(V)

	bilinear_form_inclusion = ufl.inner(sigma(eps(u), E_i, nu_i),eps(u_bar))*dx(1)
	bilinear_form_matrix = ufl.inner(sigma(eps(u), E_m, nu_m),eps(u_bar))*dx(2)
	bilinear_form = bilinear_form_inclusion + bilinear_form_matrix
	g=0.0 # no weight
	body_force = dolfinx.fem.Constant(domain, ScalarType((0,-g)))
	linear_form = ( ufl.dot(body_force,u_bar)  ) * ufl.dx

	# this finds the label of the degree of freedom for the nodes on the boundary facets
	outer_facets = facet_tags.find(2)
	#print("tags:", outer_facets)
	outer_boundary_dofs = dolfinx.fem.locate_dofs_topological(V, 1, outer_facets)
	#print("dofs:",outer_boundary_dofs)

	uD = dolfinx.fem.Function(V)
	u_on_boundary = lambda x: np.array([-x[1], -x[0]], dtype=ScalarType)
	uD.interpolate(u_on_boundary)
	bc = dolfinx.fem.dirichletbc(uD, outer_boundary_dofs)

	problem = dolfinx.fem.petsc.LinearProblem(bilinear_form, linear_form, bcs=[bc], 
											  petsc_options={"ksp_type": "preonly", "pc_type": "lu"})
	u_solution = problem.solve()

	from eshelby import EshelbyDisk
	solution = EshelbyDisk(V,R_e/R_i, E_i/E_m, nu_i, nu_m)
	u_ref_func = solution.to_function(R_i)
	
	# Shear computation
	#compute the numerical solution of the shear strain
	eps_solution = eps(u_solution) #strain tensor
	V_eps = dolfinx.fem.FunctionSpace(domain,("DG", 0))
	#shear strain
	eps_xy_expr = dolfinx.fem.Expression(eps_solution[0,1],V_eps.element.interpolation_points())
	eps_xy = dolfinx.fem.Function(V_eps)
	eps_xy.interpolate(eps_xy_expr)
	num_eps_xy = extract.solution(domain, eps_xy, 0.5, 0.3)
	
			
	#compute the analytical solution of the shear strain
	# In the case of a circular inclusion, eps_xy(x,y) should be equal to
	mu_m = E_m/(2*(1+nu_m))
	mu_i = E_i/(2*(1+nu_i))
	q = (3-4*nu_m)/(8*mu_m*(1-nu_m))
	b = 1/(1+2*q*(mu_i-mu_m))
	eps_ref=-b
	
	#compute mean values of shear over inclusion and matrix
	mean_shear_m=dolfinx.fem.assemble_scalar(dolfinx.fem.form(eps_xy * dx(2))) / area_matrix
	mean_shear_i=dolfinx.fem.assemble_scalar(dolfinx.fem.form( eps_xy *dx(1) )) / area_inclusion
	
	#compute the deviation inside the inclusion
	deviation_xy = dolfinx.fem.assemble_scalar(dolfinx.fem.form(np.absolute(eps_xy-mean_shear_i)*dx(1)))/ mean_shear_i

	return u_solution, u_ref_func,domain, num_eps_xy, eps_ref, mean_shear_m, mean_shear_i, deviation_xy

#Ns=[2.5,5,10,15,20]
Ns= np.arange(5,20,dtype=int)

deg=[1,2]

# rates=np.zeros((len(Ns)-1, len(deg)),dtype=np.float64) #each column is for a different degree order
errorsL2=np.zeros((len(Ns), len(deg)),dtype=np.float64) #each column is for a different degree order
errorsH1=np.zeros((len(Ns), len(deg)),dtype=np.float64) #each column is for a different degree order

for i,D in enumerate(deg):
	EL2 = np.zeros(len(Ns), dtype=ScalarType)
	EH1 = np.zeros(len(Ns), dtype=ScalarType)
	hs = np.zeros(len(Ns), dtype=np.float64)
#	 print(f"h: {deg[i]:.2e}")
	for k,N in enumerate(Ns):
		u_sol, u_ref,domain,num_eps_xy, eps_ref, mean_shear_m, mean_shear_i,deviation_xy=solve_eshelby(N,D)
		eh=u_sol-u_ref
		inner_gradient=(ufl.inner(ufl.grad(eh),ufl.grad(eh)))
		EH1[k] = np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form(inner_gradient * ufl.dx)))
		EL2[k] = np.sqrt(dolfinx.fem.assemble_scalar(dolfinx.fem.form( ufl.dot(eh,eh) * ufl.dx) ))
		hs[k] = R_i/Ns[k]
#		 print(f"h: {hs[k]:.2e} Error: {Es[k]:.2e}")
		#store errors for each level-line- for each degree -column-
		errorsL2[k,i]=EL2[k]
		errorsH1[k,i]=EH1[k]
	ratesL2 = np.log(EL2[1:]/EL2[:-1])/np.log(hs[1:]/hs[:-1])
	ratesH1 = np.log(EH1[1:]/EH1[:-1])/np.log(hs[1:]/hs[:-1])
	print(f"Polynomial degree {D:d}, Rates L2 {np.mean(ratesL2)}, Rates H1 {np.mean(ratesH1)}")

# plot of L2 errors in function of mesh size and lagrange FE degree
plt.loglog(hs,errorsL2[:,0],'.-b', label="Lagrange P1 - L2") #L2 error for degree 1
plt.loglog(hs,errorsH1[:,0],'.-r',label="Lagrange P1 - H1") #H1 error for degree 1

#plt.loglog(hs,errorsL2[:,1],'.-b', label="Lagrange P2 - L2") #L2 error for degree 2

plt.legend()
plt.grid()
plt.xlabel("mesh size")
plt.ylabel("Erreur")
plt.show()