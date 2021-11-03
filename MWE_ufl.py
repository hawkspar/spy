from dolfin import *
from ufl.algorithms.compute_form_data import compute_form_data

# Physical parameters
m=-1
Re=200
mesh=Mesh('Mesh/validation/validation.xml')
FE_vector=VectorElement("Lagrange",mesh.ufl_cell(),2,3)
FE_scalar=FiniteElement("Lagrange",mesh.ufl_cell(),1)
FE_TH=FunctionSpace(mesh,MixedElement([FE_vector,FE_scalar]))
# Test and trial functions
test = TestFunction(FE_TH)
trial = TrialFunction(FE_TH)
# Initial conditions
q = Function(FE_TH)
File('validation/last_baseflow.xml') >> q.vector()

# Gradient with x[0] is x, x[1] is r, x[2] is theta
def grad(v,m):
	return as_tensor([[v[0].dx(0), v[0].dx(1), m*1j*v[0]],
				 	  [v[1].dx(0), v[1].dx(1), m*1j*v[1]-v[2]],
					  [v[2].dx(0), v[2].dx(1), m*1j*v[2]+v[1]]])

def div(v,m): return v[0].dx(0) + v[1].dx(1) + m*1j*v[2]

# Navier Stokes variational form
def NS_form(m):
    u,p=split(q)
    test_u,test_m=split(test)

    #mass (variational formulation)
    M = inner(div(u,m),test_m)*dx
    #set_trace()
    #momentum (different test functions and IBP)
    M += 	  inner(grad(u,m)*u,	test_u)   *dx # Convection
    M += 1/Re*inner(grad(u,m), grad(test_u,m))*dx # Diffusion
    M -= 	  inner(p, 			div(test_u,m))*dx # Pressure
    return M

diff_form=derivative(NS_form(m),q,trial)
#compute_form_data(diff_form,complex_mode=True)
parameters['linear_algebra_backend'] = 'Eigen'
Lm=as_backend_type(assemble(diff_form,form_compiler_parameters={'complex_mode':True})).sparray()