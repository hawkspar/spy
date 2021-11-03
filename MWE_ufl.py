from ufl import *
from ufl.classes import TestFunction, TrialFunction
from ufl.algorithms.compute_form_data import compute_form_data

# Physical parameters
m=-1
Re=200
cell = triangle
FE_vector=VectorElement("Lagrange",cell,2,3)
FE_scalar=FiniteElement("Lagrange",cell,1)
FE_TH = MixedElement([FE_vector,FE_scalar])
# Test and trial functions
test = TestFunction(FE_TH)
trial = TrialFunction(FE_TH)
# Initial conditions
q = Coefficient(FE_TH)

# Gradient with x[0] is x, x[1] is r, x[2] is theta
def grad(v,m):
	return as_tensor([[v[0].dx(0), v[0].dx(1), m*1j*v[0]],
				 	  [v[1].dx(0), v[1].dx(1), m*1j*v[1]-v[2]],
					  [v[2].dx(0), v[2].dx(1), m*1j*v[2]+v[1]]])
def grad_re(v):
	return as_tensor([[v[0].dx(0), v[0].dx(1),  0],
                      [v[1].dx(0), v[1].dx(1), -v[2]],
                      [v[2].dx(0), v[2].dx(1),  v[1]]])

def div(v,m): return v[0].dx(0) + v[1].dx(1) + m*1j*v[2]

# Navier Stokes variational form
def NS_form(m):
    u,p=split(q)
    test_u,test_m=split(test)

    #mass (variational formulation)
    M = div(u,m)*test_m*dx
    #set_trace()
    #momentum (different test functions and IBP)
    M += 	    dot(grad(u,m)*u,	test_u)   *dx # Convection
    M += 1/Re*inner(grad(u,m), grad(test_u,m))*dx # Diffusion
    M -= 	    dot(p, 			div(test_u,m))*dx # Pressure
    return M

pert_form=compute_form_data(NS_form(m),complex_mode=True)