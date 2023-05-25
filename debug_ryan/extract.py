import numpy as np
import dolfinx # FEM in python

def find_cells(points,domain):
    """
    Find the cells of the mesh `domain` where the points `points` lie
    """
    bb_tree = dolfinx.geometry.BoundingBoxTree(domain, domain.topology.dim)
    cells = []
    points_on_proc = []
    # Find cells whose bounding-box collide with the the points
    cell_candidates = dolfinx.geometry.compute_collisions(bb_tree, points.T)
    # Choose one of the cells that contains the point
    colliding_cells = dolfinx.geometry.compute_colliding_cells(domain, cell_candidates, points.T)
    for i, point in enumerate(points.T):
        if len(colliding_cells.links(i))>0:
            points_on_proc.append(point)
            cells.append(colliding_cells.links(i)[0])
    points_on_proc = np.array(points_on_proc, dtype=np.float64)
    return points_on_proc, cells


def solution(domain, solu_name, xval, yval, zval=0):
    """
    gives the value of the solution at the point (xval,yval)
    """
    points = np.array([[xval],[yval],[zval]]) # dummy 3rd element
    pointsT, cells = find_cells(points,domain)
    out = solu_name.eval(pointsT, cells)
    return out 