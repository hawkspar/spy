from dolfin import *
from pdb import set_trace

# Trivial geometry
mesh = UnitSquareMesh(2,2)
r = SpatialCoordinate(mesh)[1]
# Physical parameters
m=-1
Re=200
# Taylor Hodd elements ; stable element pair
FE_vector=VectorElement("Lagrange",mesh.ufl_cell(),2,3)
FE_scalar=FiniteElement("Lagrange",mesh.ufl_cell(),1)
V=FunctionSpace(mesh,MixedElement([FE_vector,FE_scalar]))
# Test and trial functions
test = TestFunction(V)
trial = TrialFunction(V)
# Initial conditions
q = interpolate(Constant([1,0,0,0]), V)

# Gradient with x[0] is x, x[1] is r, x[2] is theta
def grad(v,m):
	return as_tensor([[v[0].dx(0), v[0].dx(1), m*1j*v[0]/r],
				 	  [v[1].dx(0), v[1].dx(1), m*1j*v[1]/r-v[2]/r],
					  [v[2].dx(0), v[2].dx(1), m*1j*v[2]/r+v[1]/r]])
def grad_re(v):
	return as_tensor([[v[0].dx(0), v[0].dx(1),  0],
				 	  [v[1].dx(0), v[1].dx(1), -v[2]/r],
					  [v[2].dx(0), v[2].dx(1),  v[1]/r]])

def div(v,m): return v[0].dx(0) + (r*v[1]).dx(1)/r + m*1j*v[2]/r

# Navier Stokes variational form
def NS_form(m):
    u,p=split(q)
    test_u,test_m=split(test)

    #mass (variational formulation)
    M = div(u,m)*test_m*r*dx
    #set_trace()
    #momentum (different test functions and IBP)
    M += 	    dot(grad(u,m)*u,	test_u)   *r*dx # Convection
    M += 1/Re*inner(grad(u,m), grad(test_u,m))*r*dx # Diffusion
    M -= 	    dot(p, 			div(test_u,m))*r*dx # Pressure
    return M
    
# Compute baseflow
solve(NS_form(0) == 0, q, [])

from ufl.algorithms.compute_form_data import compute_form_data

pert_form=compute_form_data(NS_form(m),complex_mode=True)
parameters['linear_algebra_backend'] = 'Eigen'
L = as_backend_type(assemble(derivative(pert_form,q,trial))).sparray()