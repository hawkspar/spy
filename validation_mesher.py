from mshr import Polygon, generate_mesh
from dolfin import *
from matplotlib import pyplot as plt
from matplotlib import tri as tri
import numpy as np

mesh_path="./Mesh/validation/"

yn=int((6e3/7)**.5)
#yn=10
# Generate mesh
mesh_core=RectangleMesh(Point(0,0),Point(70,10), 7*yn, yn)

mesh_core_file = File(mesh_path+"validation_core.xml")
mesh_core_file << mesh_core
"""
# Print workaround
n = mesh_core.num_vertices()
d = mesh_core.geometry().dim()

# Create the triangulation
mesh_coordinates = mesh_core.coordinates().reshape((n, d))
triangles = np.asarray([cell.entities(0) for cell in cells(mesh_core)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0],
                                  mesh_coordinates[:, 1],
                                  triangles)

# Plot the mesh
plt.figure()
plt.triplot(triangulation)
plt.savefig(mesh_path+'mesh_core.png')
"""
domain = Polygon([Point(70,0),Point(120,0),Point(120,60),Point(0,60),Point(0,10),Point(70,10)])
n=160000
#n=10
mesh_extended = generate_mesh(domain,n)

mesh_extended_file = File(mesh_path+"validation_extended.xml")
mesh_extended_file << mesh_extended
"""
# Print workaround
n = mesh_extended.num_vertices()
d = mesh_extended.geometry().dim()

# Create the triangulation
mesh_coordinates = mesh_extended.coordinates().reshape((n, d))
triangles = np.asarray([cell.entities(0) for cell in cells(mesh_extended)])
triangulation = tri.Triangulation(mesh_coordinates[:, 0],
                                  mesh_coordinates[:, 1],
                                  triangles)

# Plot the mesh
plt.figure()
plt.triplot(triangulation)
plt.savefig(mesh_path+'mesh_extended.png')
"""