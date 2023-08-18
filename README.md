# About this code

This code was developed by Quentin CHevalier as part of his PhD from 2020 to 2023.

The acronym means "Swirling Parallel Yet another jet code" as an hmomage to Chuhan Wang who gave me the first building blocks for this code.

## Capabilities - what this code can do for you

3 components axisymmetric baseflow, 3 components azimuthally decomposed perturbations.

Handles arbitrary eddy viscosity.

Eigenvalue and resolvent calculations on a cylindrical 2D mesh with azimuthal decomposition.

Parallel, handles complex numbers, baseflow computation using a Newton scheme, arbitrary 2D planar geometry and Dirichlet boundary conditions.

Contains routines for 2D or 3D visualisation.

Memoisation all around. Utilities and handlers.

## Shortfalls - what it cannot do

Current memory leak - keep an eye out when performing overly large computations !

Depends under the hood on _PETSc_ and _SLEPc_, and as capricious as them.

Non-Dirichlet, non stress-free BCs

# Installation 

This code requires _python3_.

## Dependancies

This code depends on the _FEniCsX_ library. This library is included as `dolfinx` in the code.

It is strongly recommanded to use the _docker_ image of this library available at https://hub.docker.com/r/dolfinx/dolfinx

Parts of the project are also dependant on the _meshio_ library that doesn't come in the docker image. It can be readily installed from the command line inside the container by using _pip_ : `pip3 install h5py meshio`

_Plotly_ for 3D figures has to be installed as well.

# Basic use

## Structure

The tree of the project separates `cases` which represents user application, and `src`, where most of the magic happens.

`src/helpers.py` provides a couple useful stuff like operators, convenient file finders, but `spy.py`, `spyb.py` and `spyp.py` provide only classes.

The reference case should be `nozzle`. This case uses most of the interesting features currently implemented.

Every case folder must include a `mesh` folder with the desired meshes in _xdmf_ format.

It is advised to regroup utilities in a `setup.py` which can be used to prescribe solver parameters, mesh directions, boundary conditions and boxing functions...

During computations, _SPY_ will try to create directories in your case file. It will save every result it can using complicated strings formats like `q_Re=2000_S=0,2_St=1,2e-04_n=12_p=0.npy`. Don't panic ! It knows what it's doing. If you want to go parallel, be aware that _SPY_'s saving scheme is number of processor dependent - pick a number of processors and stick to it !

## Setup

First things first - _SPY_ is direction agnostic. This means that you must tell _SPY_ which direction is which in your mesh. This is done by providing a `direction_map` to it at creation. Something like `direction_map={'x':0,'r':1,'th':2}`.

Another key step is to choose tolerances. It goes without saying that some computations are very sensitive to this, especially to the relaxation parameter for a Newton solver. This is expected to be specified as another dictionary:

```python
params = {"rp":.97,     # Relaxation parameter
		  "atol":1e-12, # Absolute tolerance
		  "rtol":1e-9,  # Relative tolerance
		  "max_iter":10}
```

Then you'll need to import _SPY_ classes like so:

```python
import sys
sys.path.append('/home/shared/src')
from spy  import SPY
from spyb import SPYB
from spyp import SPYP
```

Next step is defining your geometry. _SPY_ doesn't read anything on a mesh other than points and edges, therefore it is advised to use indicator functions to build geometry.

```python
import numpy as np
def symmetry(x:ufl.SpatialCoordinate) -> np.ndarray: return np.isclose(x[params['r']],0,params['atol']) # Axis of symmetry at r=0
```

## Computing a baseflow

In order to compute a baseflow, one must use the _SPYB_ (for "SPY Baseflow") module. This module is a child of _SPY_ and inherits all its convenience functions.

Its call signature goes like `SPYB(params,data_path,base_mesh,direction_map)` with computation parameters, path to file, name of mesh inside `mesh/`, and dictionary with directions.

Axisymmetric baseflows should be computed in real mode, running `source /usr/local/bin/dolfinx-real-mode` before anything else. This improves storage space, and performance.

### Boundary conditions

The first thing to do is define boundary conditions. The simplest of these, homogeneous boundary conditions, can be specified directly using `SPY.applyHomogeneousBCs`, providing a geometry indicator function, a list of directions and optionally a value. By defaults it sets the relevant velocity component at the provided boundary to zero.

More involved custom Dirichlet boundary conditions can be speficied using `dolfinx.fem.locate_dofs_geometrical`, `dolfinx.fem.dirichletbc` and `SPY.applyBCs`. Like all functions supplied to `dolfinx.fem.Function.interpolate`, the boundary condition must take as argument a single array containing two lines for the evaluation points and return an appropriately sized value for every point.

The following snippet gives the procedure for imposing the BC `profile` on component `dir` at `geo_indic`:

```python
subspace=spy.TH.sub(0).sub(dir)
subspace_collapsed,_=subspace.collapse()
u=Function(subspace_collapsed)
u.interpolate(profile)
# Degrees of freedom
dofs = dfx.fem.locate_dofs_geometrical((subspace, subspace_collapsed), geo_indic)
bcs  = dfx.fem.dirichletbc(u, dofs, subspace) # Same as OpenFOAM
# Actual BCs
spy.applyBCs(dofs[0],bcs)
```

`nozzle\src\baseflow` implements an additional trick - it is possible to change a boundary condition on the fly during a computation by providing `profile` as a class with a `__call__` method and changing the attributes of the `profile` object.

Right now, _SPY_ does not accomodate non-Dirichlet boundary conditions, you'll have to change the equations weak form for more.

If nothing is specified for a component on a boundary, it will default to stress-free "Pn=(grad U+grad U^T)n/Re" one.

### About eddy viscosity

_SPYB_ can handle an eddy viscosity but is unable to compute one ! The only way to include one right now is to compute one using _OpenFOAM_, convert it to _meshio_ friendly format using a custom _Paraview_ macro and feed it to `cases/nozzle/src/OpenFOAM_reader`. This will create a `baseflow` directory with a `nut` subfolder containing `.npy` files that _SPY_ can use. Note that _OpenFOAM_ and _SPY_ can operate on different meshes, as long as the _SPY_ one is included in the _OpenFOAM_ one.

#### The Paraview macro - link between OpenFOAM and SPY

```python
from paraview.simple import *

association_dict={'1000':{'Re':1000,'S':0},'2000':{'Re':10000,'S':0},'3000':{'Re':100000,'S':0},'6000':{'Re':200000,'S':0}}

for i in range(1,101): association_dict[str(6000+i*2000)]={'Re':200000, 'S':.01*i}

for vtk in association_dict:
    # find source
    nozzle_vtm = XMLMultiBlockDataReader(registrationName='nozzle_'+vtk+'.vtm', FileName=['/home/chevalier/Documents/incompressible_jets/OpenFOAM/nozzle/VTK/nozzle_'+vtk+'.vtm'])

    # create a new 'Append Location Attributes'
    appendLocationAttributes = AppendLocationAttributes(registrationName='AppendLocationAttributes', Input=nozzle_vtm)

    save_str='/home/chevalier/Documents/incompressible_jets/SPY/cases/nozzle/baseflow/OpenFOAM/Re='+str(association_dict[vtk]['Re'])+'_S='+str(round(association_dict[vtk]['S'],3)).replace('.',',')+'.xmf'
    # save data
    SaveData(save_str, proxy=appendLocationAttributes, ChooseArraysToWrite=1, PointDataArrays=['U', 'p', 'nut'],  CellDataArrays=['CellCenters', 'U', 'p', 'nut'])
    # clear the pipeline
    for name in ('nozzle_'+vtk+'.vtm','AppendLocationAttributes'):
        source = FindSource(name)
        SetActiveSource(source)
        Delete(source)
        del source

    with open(save_str, 'r+') as fp:
        # Read all lines
        lines = fp.readlines()
        fp.seek(0) # Move file pointer to the beginning of a file
        fp.truncate() # Erase the file
        # Cut out clumsy Times and Grids
        fp.writelines(lines[:3]+lines[5:12]+lines[15:37]+lines[39:])

    print("Handled nozzle_"+vtk+'.vtm')
```

### Starting the computation

It is possible to supply an initial guess to the Newton solver using a function or to use a previous computation as a starting point. To use a result of a previous computation, simply run `SPY.loadBaseflow` (notice that since _SPYB_ is a child of _SPY_, it inherits this method) with the required parameters. _SPYB_ will try to load the closest run to the parameters you provide according to a L1 norm - check the printed output that it got it right.

`SPYB.Re` must be set by hand prior to computation - it may be a number or a `dolfinx.fem.Function`, allowing easy implementation of sponge zones.

`SPYB.baseflow` computes the baseflow. Right now, parameters `Re` and `S` are only used in the saving string scheme. It might cause problems to have a non-integer Re. `refinement` should not be used. `save` means that the baseflow must be written to file - after all, maybe the user would want to keep it in the `SPYB` object ? `baseflow_init` allow for the velocities of the baseflow to be initialised by an arbitrary provided function.

A useful tool is `SPYB.smoothenU` that uses a pseudo-Laplace equation to smoothen out a velocity - it can handle a component or the whole vector. It should only ever be called with small `e` and operates in place on `SPY.Q` where the baseflow is to be found.

`SPY.saveBaseflow` should handle all the loading for you. It can optionally save eddy viscosity and velocity fields in `xdmf` format - currently, `FEniCsX` can write `xdmf` in parallel not read them.

`SPY.sanityCheck` and `SPY.printStuff` are good places to start debugging.

### Post processing baseflows

A couple `SPYB.computeX` methods exist to produce 3D _Plotly_ objects that can be put in relation with _SPYP_ 3D visualisations.

## Doing linear stability

This requires going complex with `source /usr/local/bin/dolfinx-complex-mode` and use of the _SPYP_ object, (for "SPY Perturbations").

The call signature is identicval to _SPYB_. It can operate on a different mesh than _SPYB_, but then its mesh has to be included in the _SPYB_ mesh.

/!\ Reading meshes in parallel _FEniCsX_ is an order-dependant process ! That means that extra care hould be taken to always load meshes in the same order - for instance initialising _SPYB_ in the `setup` file so that it is always read first !

Boundary conditions are applied like before.

### Using a baseflow

Use `SPYP.interpolateBaseflow` given a `SPYB` object to accomodate different meshes more easily.

### Launching the computations

For performance reasons and to accomodate loops, it is necessary to assemble matrices prior a run. Use `SPYP.assembleNMatrix` for the mass one, and `SPYP.assembleJMatrix` for the linearised Navier-Stokes equations given an `m`.

Eigenvalue computations are performed using `SPYP.eigenvalues` around a provided shift `sigma`, in number `k`. Again, `Re`, `S` and `m` are purely for saving purposes.

To read the eigenvalues and plot them one should use `cases/nozzle/src/print/eig.py`.

## Doing resolvent analysis

The process is very similar to linear stability. The only difference is that an additional routine should be used prior launch `SPYP.assembleMRMatrices` where an indicator function can be used to constrain forcing.

Launch is done using `SPYP.resolvent` specifying number of required modes, list of frequencies and again `Re`, `S` and `m` to print saving strings.

### Post processing resolvent analysis

There are more options to print modes. Loading modes is easy with `SPYP.readMode`, 2D cuts are named `SPYP.saveX` and 3D stuff `SPYP.computeX`. Inspiration on how to use these can be found in `cases/nozzle/src/print/`

Careful using isosurfaces, _Plotly_ is capricious and can only handle regular meshes - i.e. stuff from `numpy.meshgrid`.