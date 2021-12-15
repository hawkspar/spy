import meshio #pip3 install --no-cache-dir --no-binary=h5py h5py meshio

def create_mesh(mesh, cell_type):
    cells = mesh.get_cells_type(cell_type)
    cell_data = mesh.get_cell_data("gmsh:physical", cell_type)
    out_mesh = meshio.Mesh(points=mesh.points[:, :2], cells={cell_type: cells}, cell_data={"name_to_read": [cell_data]})
    return out_mesh

msh = meshio.read("Mesh/validation/validation.msh")
# Create and save one file for the mesh, and one file for the facets 
triangle_mesh = create_mesh(msh, "triangle")
meshio.write("Mesh/validation/validation.xdmf", triangle_mesh)