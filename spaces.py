from dolfin import *

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True

mesh = Mesh()
with XDMFFile("Mesh/Mesh.xdmf") as infile:
    infile.read(mesh)

###############################################################################

R=FiniteElement("Lagrange",mesh.ufl_cell(),2)
V=VectorElement("Lagrange",mesh.ufl_cell(),2)
Q=FiniteElement("Lagrange",mesh.ufl_cell(),1)
TH = MixedElement([R, V, Q])
X=FunctionSpace(mesh,TH)
Urho=FunctionSpace(mesh,R)
Uv=FunctionSpace(mesh,V)
Up=FunctionSpace(mesh,Q)
