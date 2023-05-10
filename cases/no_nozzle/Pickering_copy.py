# trace generated using paraview version 5.10.0-RC1

#### import the simple module from the paraview
from paraview.simple import *

ms=[0,1,3]
Sts=['0,05','0,20','0,60']
name_tups=[('response','r'),('forcing','f')]

for m in ms:
    for  St in Sts:
        for tup in name_tups:
            # Open file
            xdmf = Xdmf3ReaderS(registrationName=f'{tup[1]}_Re=400000_S=0,0_m={m}_St={St}_i=1.xdmf', FileName=[f'/home/chevalier/Documents/incompressible_jets/SPY/cases/no_nozzle/resolvent/{tup[0]}/print/{tup[1]}_Re=400000_S=0,0_m={m}_St={St}_i=1.xdmf'])
            source_xdmf = FindSource(f'{tup[1]}_Re=400000_S=0,0_m={m}_St={St}_i=1.xdmf')
            render = GetActiveViewOrCreate('RenderView')
            xdmfDisplay = Show(xdmf, render, 'UnstructuredGridRepresentation')
            xdmfDisplay.Representation = 'Surface'
            render.Update()
            # Only look at axial direction
            ColorBy(xdmfDisplay, ('POINTS', 'real_f', 'X'))
            xdmfDisplay.RescaleTransferFunctionToDataRange(True, False)
            xdmfDisplay.SetScalarBarVisibility(render, True)
            # Colorbar
            real_fLUT = GetColorTransferFunction('real_f')
            real_fLUT.ApplyPreset('Cold and Hot', True)
            real_fLUT.RGBPoints[1:7] = [0., 1., 1., 0., 0., 0.]
            real_fLUT.RGBPoints[-3:] = [1.,1.,0.]
            # Scaling
            xdmfDisplay.Scale = [.5, .25, 1.]
            xdmfDisplay.DataAxesGrid.Scale = [.5, .25, 1.]
            render.AxesGrid.Visibility = 1
            render.AxesGrid.DataScale = [1., .5, 1.]
            # Renaming
            render.AxesGrid.XTitle = 'x/D'
            render.AxesGrid.YTitle = 'r/D'
            real_fLUTColorBar = GetScalarBar(real_fLUT, render)
            # Colorbar placement
            real_fLUTColorBar.WindowLocation = 'Any Location'
            real_fLUTColorBar.Orientation = 'Horizontal'
            real_fLUTColorBar.Position = [0.06846429011964988, 0.5060134003350083]
            real_fLUTColorBar.ScalarBarLength = 0.3299999999999998
            # Snapshot
            layout = GetLayout()
            layout.SetSize(2211, 1194)
            render.InteractionMode = '2D'
            render.CameraPosition = [12.800000011920929, 1.25, 47.38395384515032]
            render.CameraFocalPoint = [12.800000011920929, 1.25, 0.0]
            render.CameraParallelScale = 7.2718213082876675
            SaveScreenshot(f'/home/chevalier/Documents/incompressible_jets/SPY/cases/no_nozzle/colormaps/Pickering_{tup[1]}_copy_m={m}_St={St}.png', render, ImageResolution=[2211, 1194])
